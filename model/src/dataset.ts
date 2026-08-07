/** Dataset loader and deterministic synthetic negative generation. */

import type { CaptureExport, LabeledSample, TrajectoryPoint } from '../../src/types.js';
import { extractFeatureVector } from './features.js';
import { RNG } from '../../src/rng.js';
import { HARD_NEGATIVE_GENERATORS } from './hard-negatives.js';

export type TrainingSample = LabeledSample & { groupId: string; calibration?: true };

export interface BuildDatasetOptions {
  seed?: number;
  /** Explicit operator attestation permitting legacy v1 capture records. */
  allowLegacy?: boolean;
}

function finite(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

/**
 * Trailing samples where the cursor has stopped moving.
 *
 * The extension appends the click position after motion ends, so 99.4% of real captured
 * trajectories finish with at least one duplicate position. That makes the final-frame
 * speed exactly zero, and `extractFeatureVector` feature 8 (finalSpeedRatio) therefore
 * reads ~0 for 99.6% of real trajectories while every generator terminates with residual
 * velocity. A single threshold on that one feature separated real from synthetic at
 * 99.9% — the model was detecting the CAPTURE PIPELINE, not the motion, which is why
 * training hit 100% validation accuracy within one epoch.
 *
 * Trimming the stationary tail puts both classes on the same convention so the features
 * describe movement rather than provenance.
 */
function trimStationaryTail(path: TrajectoryPoint[]): TrajectoryPoint[] {
  let end = path.length;
  while (end > 2) {
    const last = path[end - 1]!;
    const previous = path[end - 2]!;
    if (last.x !== previous.x || last.y !== previous.y) break;
    end--;
  }
  return end === path.length ? path : path.slice(0, end);
}

function normalizedTrajectory(path: unknown, legacy: boolean): TrajectoryPoint[] | null {
  if (!Array.isArray(path)) return null;
  const clean = path
    .filter((point): point is TrajectoryPoint => point !== null && typeof point === 'object' &&
      finite((point as TrajectoryPoint).x) && finite((point as TrajectoryPoint).y) && finite((point as TrajectoryPoint).tMs))
    .sort((a, b) => a.tMs - b.tMs);
  // A human trajectory is not invalid merely because it is short, slow, or
  // sparsely sampled. Two finite samples at distinct times are sufficient for
  // the canonical feature extractor; richer features naturally fall back to
  // their finite neutral values when the path is too short to measure them.
  if (clean.length < 2) return null;

  let start = 0;
  if (legacy) {
    // v1 buffered all mousemoves since the prior click. Keep only the final
    // contiguous active burst so long idle gaps do not become fake motion.
    for (let index = 1; index < clean.length; index++) {
      if (clean[index]!.tMs - clean[index - 1]!.tMs > 250) start = index;
    }
  }
  const burst = trimStationaryTail(clean.slice(start));
  if (burst.length < 2) return null;
  const origin = burst[0]!.tMs;
  const normalized = burst.map((point) => ({ x: point.x, y: point.y, tMs: point.tMs - origin }));
  const duration = normalized[normalized.length - 1]!.tMs;
  // Duration limits used to censor legitimate human hesitations and very fast
  // corrections. Only a non-positive time span is structurally unmeasurable.
  return duration > 0 ? normalized : null;
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

// Soma's own output is deliberately NOT part of this dataset. It was previously
// injected as label 0 (human) into the training split, which made the model's verdict
// on Soma circular: the scorer was taught that Soma is human, then used downstream as
// evidence that Soma is human. It also dragged the human boundary toward Soma's
// distribution, which is why real captured humans scored as substantially bot-like.
// Soma is now evaluated only as a held-out probe (see evaluateSomaPlans in train.ts),
// so "Soma scores as human" is a measurement rather than a construction.

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
      // A v2 export can still carry individual v1-era movement records. Those lack the
      // trustedEvents provenance marker, so without an explicit legacy attestation they
      // are SKIPPED rather than aborting the run: dropping unattested records is the
      // conservative reading, and a mixed export is otherwise unusable.
      if (legacy && !allowLegacy) continue;
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

  // Negative classes. Two tiers, both labeled 1 (machine):
  //
  //   EASY  — straight line, constant velocity, white noise, naive ease, random
  //           waypoints. Separable on curvature or tremor alone.
  //   HARD  — faithful reimplementations of the humanization libraries a real
  //           detector is trained against (ghost-cursor, bezmouse, HumanCursor).
  //           These already carry Bezier curvature, Fitts-like duration, and
  //           jitter, so they sit near the actual decision boundary.
  //
  // Training on easy negatives alone yields a model that separates them at ~99.8%
  // while scoring real humans as substantially bot-like: it learns "not-easy-synth"
  // rather than "human". The hard tier is what forces the boundary to the right place.
  const easyCount = Math.max(capturedHumanCount, 100);
  for (let i = 0; i < easyCount; i++) {
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

  // Hard negatives, sized to match the human class so the boundary is not dominated
  // by the trivially separable tier. Each library gets its own groupId namespace so
  // the group-disjoint split cannot leak one library between train and test.
  const hardPerLibrary = Math.ceil(Math.max(capturedHumanCount, 100) / HARD_NEGATIVE_GENERATORS.length);
  for (const { source, generate } of HARD_NEGATIVE_GENERATORS) {
    for (let i = 0; i < hardPerLibrary; i++) {
      const start = { x: rng.nextRange(20, 1500), y: rng.nextRange(20, 900) };
      const end = { x: rng.nextRange(20, 1500), y: rng.nextRange(20, 900) };
      if (Math.hypot(end.x - start.x, end.y - start.y) < 20) continue;
      samples.push({
        features: extractFeatureVector(generate(start, end, rng)),
        label: 1,
        source,
        groupId: `hard-${source}-${Math.floor(i / 20)}`,
      });
    }
  }

  for (let i = samples.length - 1; i > 0; i--) {
    const j = rng.nextInt(i + 1);
    [samples[i], samples[j]] = [samples[j]!, samples[i]!];
  }
  return samples;
}
