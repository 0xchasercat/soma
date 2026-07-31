/** Dataset loader and deterministic synthetic negative generation. */

import type { CaptureExport, LabeledSample, TrajectoryPoint } from '../../src/types.js';
import { extractFeatureVector } from './features.js';
import { RNG } from '../../src/rng.js';
import { synthesizeMovement } from '../../src/pointer/trajectory.js';

export type TrainingSample = LabeledSample & { groupId: string; calibration?: true };

export interface BuildDatasetOptions {
  seed?: number;
  /** Explicit operator attestation permitting legacy v1 capture records. */
  allowLegacy?: boolean;
}

function finite(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function normalizedTrajectory(path: unknown, legacy: boolean): TrajectoryPoint[] | null {
  if (!Array.isArray(path)) return null;
  const clean = path
    .filter((point): point is TrajectoryPoint => point !== null && typeof point === 'object' &&
      finite((point as TrajectoryPoint).x) && finite((point as TrajectoryPoint).y) && finite((point as TrajectoryPoint).tMs))
    .sort((a, b) => a.tMs - b.tMs);
  if (clean.length < 10) return null;

  let start = 0;
  if (legacy) {
    // v1 buffered all mousemoves since the prior click. Keep only the final
    // contiguous active burst so long idle gaps do not become fake motion.
    for (let index = 1; index < clean.length; index++) {
      if (clean[index]!.tMs - clean[index - 1]!.tMs > 250) start = index;
    }
  }
  const burst = clean.slice(start);
  if (burst.length < 10) return null;
  const origin = burst[0]!.tMs;
  const normalized = burst.map((point) => ({ x: point.x, y: point.y, tMs: point.tMs - origin }));
  const duration = normalized[normalized.length - 1]!.tMs;
  return duration >= 50 && duration <= 10_000 ? normalized : null;
}

/** Load and minimally validate a capture export before training. */
export async function loadCapturedData(jsonPath: string): Promise<CaptureExport> {
  const json = await Bun.file(jsonPath).json() as Partial<CaptureExport>;
  if (
    (json.schema !== 'soma.capture.v1' && json.schema !== 'soma.capture.v2') ||
    !Array.isArray(json.movements) ||
    !Array.isArray(json.keystrokes) ||
    !Array.isArray(json.scrolls)
  ) {
    throw new Error(`Invalid capture export ${jsonPath}: expected soma.capture.v1/v2 with movements, keystrokes, and scrolls arrays`);
  }
  return json as CaptureExport;
}

function generateStraightLine(start: { x: number; y: number }, end: { x: number; y: number }, durationMs: number): TrajectoryPoint[] {
  const path: TrajectoryPoint[] = [];
  const steps = 20;
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    path.push({ x: start.x + (end.x - start.x) * t, y: start.y + (end.y - start.y) * t, tMs: t * durationMs });
  }
  return path;
}

function generateConstantVelocity(start: { x: number; y: number }, end: { x: number; y: number }, durationMs: number): TrajectoryPoint[] {
  const path: TrajectoryPoint[] = [];
  const steps = 30;
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    path.push({ x: start.x + (end.x - start.x) * t, y: start.y + (end.y - start.y) * t, tMs: t * durationMs });
  }
  return path;
}

function generateWhiteNoise(start: { x: number; y: number }, end: { x: number; y: number }, rng: RNG, durationMs: number): TrajectoryPoint[] {
  const path: TrajectoryPoint[] = [];
  const steps = 40;
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    path.push({
      x: start.x + (end.x - start.x) * t + rng.nextGaussian(0, 2),
      y: start.y + (end.y - start.y) * t + rng.nextGaussian(0, 2),
      tMs: t * durationMs,
    });
  }
  return path;
}

function generateNaiveEase(start: { x: number; y: number }, end: { x: number; y: number }, durationMs: number): TrajectoryPoint[] {
  const steps = 36;
  return Array.from({ length: steps + 1 }, (_, index) => {
    const t = index / steps;
    const eased = t * t * (3 - 2 * t);
    return {
      x: start.x + (end.x - start.x) * eased,
      y: start.y + (end.y - start.y) * eased,
      tMs: t * durationMs,
    };
  });
}

function generateRandomWaypoints(
  start: { x: number; y: number },
  end: { x: number; y: number },
  rng: RNG,
  durationMs: number,
): TrajectoryPoint[] {
  const points = [
    start,
    { x: start.x + (end.x - start.x) * 0.35 + rng.nextRange(-50, 50), y: start.y + (end.y - start.y) * 0.35 + rng.nextRange(-50, 50) },
    { x: start.x + (end.x - start.x) * 0.7 + rng.nextRange(-40, 40), y: start.y + (end.y - start.y) * 0.7 + rng.nextRange(-40, 40) },
    end,
  ];
  const path: TrajectoryPoint[] = [];
  const segmentSteps = 10;
  for (let segment = 0; segment < points.length - 1; segment++) {
    for (let step = segment === 0 ? 0 : 1; step <= segmentSteps; step++) {
      const t = step / segmentSteps;
      const from = points[segment]!;
      const to = points[segment + 1]!;
      path.push({
        x: from.x + (to.x - from.x) * t,
        y: from.y + (to.y - from.y) * t,
        tMs: ((segment * segmentSteps + step) / ((points.length - 1) * segmentSteps)) * durationMs,
      });
    }
  }
  return path;
}

function generateSomaCalibration(seed: number, count: number): TrainingSample[] {
  const rng = new RNG(seed ^ 0x51a7cafe);
  const samples: TrainingSample[] = [];
  for (let index = 0; index < count; index++) {
    const start = { x: rng.nextRange(20, 900), y: rng.nextRange(20, 650) };
    const target = {
      x: rng.nextRange(20, 900),
      y: rng.nextRange(20, 650),
      width: rng.nextRange(24, 180),
      height: rng.nextRange(20, 90),
    };
    const plan = synthesizeMovement(start, target, undefined, seed + 100_000 + index);
    samples.push({
      features: extractFeatureVector(plan.points),
      label: 0,
      source: 'soma-calibration',
      groupId: `soma-calibration-${Math.floor(index / 20)}`,
      calibration: true,
    });
  }
  return samples;
}

/** Build deterministic human and synthetic samples, retaining session groups. */
export function buildDataset(
  captured: CaptureExport | readonly CaptureExport[],
  seedOrOptions: number | BuildDatasetOptions = {},
  explicitOptions: Omit<BuildDatasetOptions, 'seed'> | boolean = {},
): TrainingSample[] {
  const captures = Array.isArray(captured) ? captured : [captured];
  const seed = typeof seedOrOptions === 'number' ? seedOrOptions : seedOrOptions.seed ?? 42;
  const allowLegacy = typeof seedOrOptions === 'number'
    ? (typeof explicitOptions === 'boolean' ? explicitOptions : explicitOptions.allowLegacy ?? false)
    : seedOrOptions.allowLegacy ?? false;
  const samples: TrainingSample[] = [];
  const rng = new RNG(seed);

  for (const capture of captures) {
    if (capture.schema === 'soma.capture.v1' && !allowLegacy) {
      throw new Error('Legacy v1 capture requires explicit allowLegacy operator attestation');
    }
    for (const movement of capture.movements) {
      const sourceSchema = 'sourceSchema' in movement && movement.sourceSchema !== undefined
        ? movement.sourceSchema
        : capture.schema;
      if (sourceSchema !== 'soma.capture.v1' && sourceSchema !== 'soma.capture.v2') {
        throw new Error(`Unsupported movement sourceSchema: ${String(sourceSchema)}`);
      }
      const legacy = capture.schema === 'soma.capture.v1' || sourceSchema === 'soma.capture.v1';
      if (legacy && !allowLegacy) {
        throw new Error('Legacy v1 capture requires explicit allowLegacy operator attestation');
      }
      if (!legacy && movement.trustedEvents !== true) continue;
      const trajectory = normalizedTrajectory(movement.trajectory, legacy);
      if (!trajectory) continue;
      samples.push({
        features: extractFeatureVector(trajectory),
        label: 0,
        source: legacy ? 'human-v1-attested' : 'human-v2-trusted',
        groupId: movement.sessionId || `human-${samples.length}`,
      });
    }
  }

  if (samples.length === 0) {
    throw new Error('Training requires at least one valid human trajectory; no usable movements were found');
  }

  const capturedHumanCount = samples.length;
  const syntheticCount = Math.max(capturedHumanCount, 100);
  for (let i = 0; i < syntheticCount; i++) {
    const start = { x: rng.nextRange(100, 500), y: rng.nextRange(100, 500) };
    const end = { x: rng.nextRange(100, 800), y: rng.nextRange(100, 600) };
    const synthType = rng.nextInt(5);
    const durationMs = Math.exp(rng.nextRange(Math.log(50), Math.log(10_000)));
    let trajectory: TrajectoryPoint[];
    let source: string;
    if (synthType === 0) {
      trajectory = generateStraightLine(start, end, durationMs);
      source = 'straight_line';
    } else if (synthType === 1) {
      trajectory = generateConstantVelocity(start, end, durationMs);
      source = 'constant_velocity';
    } else if (synthType === 2) {
      trajectory = generateWhiteNoise(start, end, rng, durationMs);
      source = 'white_noise';
    } else if (synthType === 3) {
      trajectory = generateNaiveEase(start, end, durationMs);
      source = 'naive_ease';
    } else {
      trajectory = generateRandomWaypoints(start, end, rng, durationMs);
      source = 'random_waypoints';
    }
    samples.push({
      features: extractFeatureVector(trajectory),
      label: 1,
      source,
      groupId: `synthetic-${source}-${Math.floor(i / 20)}`,
    });
  }

  const calibrationCount = Math.min(400, Math.max(100, Math.ceil(Math.sqrt(capturedHumanCount) * 10)));
  samples.push(...generateSomaCalibration(seed, calibrationCount));

  for (let i = samples.length - 1; i > 0; i--) {
    const j = rng.nextInt(i + 1);
    [samples[i], samples[j]] = [samples[j]!, samples[i]!];
  }
  return samples;
}
