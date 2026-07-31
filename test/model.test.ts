import { describe, expect, mock, test } from 'bun:test';
import { onnx } from 'onnx-proto';
import { buildDataset } from '../model/src/dataset.js';
import { extractFeatureVector, heuristicScore, loadJSONModel, loadJSONModelText, scoreTrajectory } from '../src/score/index.js';
import { loadONNXModel } from '../src/score/onnx.js';
import type { CaptureExport } from '../src/types.js';

function capture(
  schema: 'soma.capture.v1' | 'soma.capture.v2',
  trustedEvents?: true,
  sourceSchema = schema,
): CaptureExport {
  const trajectory = Array.from({ length: 12 }, (_, index) => ({
    x: index * 4,
    y: index * 2,
    tMs: index * 16.67,
  }));
  return {
    schema,
    schemaVersion: schema === 'soma.capture.v2' ? 2 : 1,
    exportedAt: new Date(0).toISOString(),
    userAgent: 'test',
    movements: [{
      sessionId: 'session-a',
      capturedAt: 1,
      sourceSchema,
      ...(trustedEvents ? { trustedEvents } : {}),
      trajectory,
      clickTarget: { tagName: 'BUTTON', isInteractive: true, boundingBox: { x: 40, y: 20, width: 40, height: 20 } },
      viewport: { width: 800, height: 600 },
    }],
    keystrokes: [],
    scrolls: [],
  };
}

function constantModel(logit: number) {
  const matrix = (rows: number, columns: number): number[][] =>
    Array.from({ length: rows }, () => Array<number>(columns).fill(0));
  return {
    fc1_w: matrix(104, 12), fc1_b: Array<number>(104).fill(0),
    fc2_w: matrix(64, 104), fc2_b: Array<number>(64).fill(0),
    fc3_w: matrix(32, 64), fc3_b: Array<number>(32).fill(0),
    fc4_w: matrix(1, 32), fc4_b: [logit],
  };
}

function canonicalModel(logit: number) {
  return {
    ...constantModel(logit),
    schema: 'soma.motion-model.v1',
    featureSchema: 'soma.motion-features.v2',
  };
}

describe('model artifacts and capture cohorts', () => {
  test('gates legacy records and preserves each capture schema', () => {
    const v1 = capture('soma.capture.v1', undefined, 'soma.capture.v1');
    const v2 = capture('soma.capture.v2', true, 'soma.capture.v2');
    expect(() => buildDataset(v1)).toThrow('explicit allowLegacy');
    expect(() => buildDataset(capture('soma.capture.v2'))).toThrow('no usable movements');

    const forwardDataset = buildDataset([v1, v2], { allowLegacy: true });
    const forwardSources = forwardDataset
      .filter((sample) => sample.label === 0 && sample.calibration !== true)
      .map((sample) => sample.source)
      .sort();
    const reverseSources = buildDataset([v2, v1], 42, { allowLegacy: true })
      .filter((sample) => sample.label === 0 && sample.calibration !== true)
      .map((sample) => sample.source)
      .sort();
    expect(forwardSources).toEqual(['human-v1-attested', 'human-v2-trusted']);
    expect(reverseSources).toEqual(forwardSources);
    expect(forwardDataset.filter((sample) => sample.calibration === true).length).toBeGreaterThanOrEqual(100);

    const migratedV1 = capture('soma.capture.v2', undefined, 'soma.capture.v1');
    expect(() => buildDataset(migratedV1)).toThrow('explicit allowLegacy');
    expect(buildDataset(migratedV1, { allowLegacy: true }).find((sample) => sample.label === 0 && sample.calibration !== true)?.source)
      .toBe('human-v1-attested');
  });
  test('uses path-only features and JSON then ONNX then heuristic scoring', async () => {
    const path = capture('soma.capture.v2', true).movements[0]!.trajectory;
    const features = extractFeatureVector(path);
    expect(extractFeatureVector(path, -10_000, 8_000, 1)).toEqual(features);
    expect(extractFeatureVector(path, 20, 10, 2_000)).toEqual(features);
    expect(features[8]).toBeCloseTo(1, 10);

    expect(loadJSONModelText('{}')).toBe(false);
    expect(await scoreTrajectory(path)).toBe(heuristicScore(features));

    mock.module('onnxruntime-node', () => ({
      InferenceSession: {
        create: async () => ({
          inputNames: ['input'],
          outputNames: ['output'],
          run: async () => ({ output: { data: new Float32Array([0.25]) } }),
        }),
      },
      Tensor: class {},
    }));
    const invalidBytes = new Uint8Array(await Bun.file('model/model.onnx').arrayBuffer());
    const invalidModel = onnx.ModelProto.decode(invalidBytes);
    const schemaEntry = invalidModel.metadataProps?.find((entry) => entry.key === 'soma.feature_schema');
    if (schemaEntry) schemaEntry.value = 'soma.motion-features.v1';
    const invalidPath = '/tmp/soma-invalid-feature-schema.onnx';
    await Bun.write(invalidPath, onnx.ModelProto.encode(invalidModel).finish());
    expect(await loadONNXModel(invalidPath)).toBe(false);
    expect(await loadONNXModel('model/model.onnx')).toBe(true);
    expect(loadJSONModelText(JSON.stringify(canonicalModel(Math.log(4))))).toBe(true);
    expect(await scoreTrajectory(path)).toBeCloseTo(0.8, 12);
    expect(loadJSONModelText(JSON.stringify(constantModel(Math.log(4))))).toBe(false);
    expect(await scoreTrajectory(path)).toBe(0.25);
    expect(loadJSONModelText('{}')).toBe(false);
  });

  test('bundled ONNX graph has the canonical interface', async () => {
    const bytes = new Uint8Array(await Bun.file('model/model.onnx').arrayBuffer());
    const model = onnx.ModelProto.decode(bytes);
    expect(model.graph?.input[0]?.name).toBe('input');
    expect(model.graph?.output[0]?.name).toBe('output');
    expect(model.graph?.node).toHaveLength(12);
    expect(model.graph?.initializer).toHaveLength(8);
    expect(Number(model.opsetImport[0]?.version)).toBe(13);
    expect(model.metadataProps?.find((entry) => entry.key === 'soma.feature_schema')?.value)
      .toBe('soma.motion-features.v2');
  });
  test('bundled metrics record the path-only promotion gate', async () => {
    const metrics = await Bun.file('model/model.metrics.json').json() as {
      schema: string;
      featureSchema: string;
      captureFiles: string[];
      dataset: { capturedHuman: number; somaCalibration: number; bot: number };
      selfConsistency: { botRate: number; count: number };
      negativeConsistency: { botRate: number; count: number };
      test: { accuracy: number };
    };
    expect(metrics.schema).toBe('soma.model-evaluation.v1');
    expect(metrics.featureSchema).toBe('soma.motion-features.v2');
    expect(metrics.captureFiles.every((path) => !path.includes('/') && !path.includes('\\'))).toBe(true);
    expect(metrics.dataset.somaCalibration).toBeGreaterThan(0);
    expect(metrics.selfConsistency.count).toBe(100);
    expect(metrics.selfConsistency.botRate).toBeLessThanOrEqual(0.05);
    expect(metrics.negativeConsistency.count).toBe(100);
    expect(metrics.negativeConsistency.botRate).toBeGreaterThanOrEqual(0.95);
    expect(await loadJSONModel('model/model.json')).toBe(true);
    const durationMs = 10_000;
    const steps = 600;
    const longStraight = Array.from({ length: steps + 1 }, (_, index) => ({
      x: index * 0.5,
      y: 0,
      tMs: durationMs * index / steps,
    }));
    expect(await scoreTrajectory(longStraight)).toBeGreaterThanOrEqual(0.5);
    expect(metrics.test.accuracy).toBeGreaterThanOrEqual(0.9);
  });
});
