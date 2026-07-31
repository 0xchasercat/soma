/**
 * Optional explicit ONNX inference with canonical schema/interface validation.
 * JSON remains the default portable format; ONNX uses an optional native peer.
 */

import type { ModelFeatureVector } from '../types.js';
import type * as OnnxProtoModule from 'onnx-proto';

interface InferenceSessionType {
  inputNames: readonly string[];
  outputNames: readonly string[];
  run(feeds: Record<string, unknown>): Promise<{ output?: { data: ArrayLike<number> } }>;
}

interface ValueInfoLike {
  type?: { tensorType?: { shape?: { dim?: Array<{ dimValue?: unknown }> | null } | null } | null } | null;
}

let onnxSession: InferenceSessionType | null = null;
let onnxAvailable = false;

async function validateONNXArtifact(modelPath: string): Promise<boolean> {
  // ONNX is a Node-only optional path; a runtime-selected specifier keeps parser/native dependencies out of browser bundles.
  const parserModule = ['onnx', 'proto'].join('-');
  const onnxModule = await import(parserModule) as typeof OnnxProtoModule;
  const fs = await import('node:fs/promises');
  const { onnx } = onnxModule;
  const bytes = new Uint8Array(await fs.readFile(modelPath));
  const model = onnx.ModelProto.decode(bytes);
  const featureSchema = model.metadataProps?.find((entry) => entry.key === 'soma.feature_schema')?.value;
  const input = model.graph?.input;
  const output = model.graph?.output;
  const shape = (value: ValueInfoLike | undefined): number[] =>
    value?.type?.tensorType?.shape?.dim?.map((dimension) => Number(dimension.dimValue)) ?? [];
  return featureSchema === 'soma.motion-features.v2' && input?.length === 1 && output?.length === 1 &&
    input[0]?.name === 'input' && output[0]?.name === 'output' &&
    shape(input[0]).join(',') === '1,12' && shape(output[0]).join(',') === '1,1';
}

/** Explicitly load a real ONNX model. Missing runtime or invalid files return false. */
export async function loadONNXModel(modelPath: string): Promise<boolean> {
  onnxSession = null;
  onnxAvailable = false;
  try {
    if (!await validateONNXArtifact(modelPath)) return false;
    // The native runtime is an optional platform-specific peer dependency.
    const ort = (await import('onnxruntime-node')) as {
      InferenceSession: { create(path: string): Promise<InferenceSessionType> };
    };
    const session = await ort.InferenceSession.create(modelPath);
    if (session.inputNames.length !== 1 || session.inputNames[0] !== 'input' ||
        session.outputNames.length !== 1 || session.outputNames[0] !== 'output') return false;
    onnxSession = session;
    onnxAvailable = true;
    return true;
  } catch {
    return false;
  }
}

/**
 * Run inference on a feature vector.
 *
 * @param features 12-dimensional feature vector.
 * @returns Bot probability [0, 1], or null if model unavailable.
 */
export async function runONNXInference(features: ModelFeatureVector): Promise<number | null> {
  if (!onnxAvailable || !onnxSession) return null;

  try {
    // Prepare input tensor (shape: [1, 12]).
    const ort = (await import('onnxruntime-node')) as {
      Tensor: new (type: string, data: ModelFeatureVector, dims: number[]) => unknown;
    };
    const inputTensor = new ort.Tensor('float32', features, [1, 12]);

    // Run inference.
    const feeds = { input: inputTensor };
    const results = await onnxSession.run(feeds);

    const output = results.output?.data[0];
    return typeof output === 'number' && Number.isFinite(output) && output >= 0 && output <= 1 ? output : null;
  } catch {
    return null;
  }
}

/**
 * Check if ONNX model is loaded and ready.
 */
export function isONNXAvailable(): boolean {
  return onnxAvailable;
}
