/**
 * Complete pointer trajectory synthesis.
 *
 * Composes: Fitts timing → Bezier path → overshoot → tremor → 60 Hz resampling.
 *
 * The output TrajectoryPlan is the complete synthesis result ready to dispatch
 * via CDP Input.dispatchMouseEvent or equivalent trusted-event channel.
 */

import type { Point2D, TargetBox, BehaviorProfile, TrajectoryPlan, TrajectoryPoint } from '../types.js';
import { RNG } from '../rng.js';
import { mergeProfile } from '../profile.js';
import { fittsMovementTime, dist2D, sampleEndpoint, preMoveSettle } from './fitts.js';
import { generateBezier, addOvershoot, resamplePath } from './bezier.js';
import { addTremor } from './tremor.js';

/**
 * Synthesize a complete pointer movement from start to target.
 *
 * @param start   Starting cursor position (viewport coordinates).
 * @param target  Target bounding box (already resolved via DOM.getBoxModel or equivalent).
 * @param profile Behavior profile (optional — uses DEFAULT_PROFILE if omitted).
 * @param seed    RNG seed (optional — random if omitted).
 * @returns Complete trajectory plan ready to dispatch.
 */
export function synthesizeMovement(
  start: Point2D,
  target: TargetBox,
  profile?: Partial<BehaviorProfile>,
  seed?: number,
): TrajectoryPlan {
  const finite = (value: number, fallback: number) => Number.isFinite(value) ? value : fallback;
  const safeStart: Point2D = { x: finite(start.x, 0), y: finite(start.y, 0) };
  const safeTarget: TargetBox = {
    x: finite(target.x, 0),
    y: finite(target.y, 0),
    width: Math.max(1, finite(target.width, 1)),
    height: Math.max(1, finite(target.height, 1)),
  };
  const p = mergeProfile(profile);
  const rng = new RNG(seed);
  const precision = Number.isFinite(p.precision) ? Math.max(0, Math.min(1, p.precision)) : 0.75;
  const endpoint = sampleEndpoint(safeTarget, precision, rng);
  const distance = dist2D(safeStart, endpoint);
  const targetWidth = Math.min(safeTarget.width, safeTarget.height);
  const speed = Number.isFinite(p.speed) && p.speed > 0 ? p.speed : 1;
  const mt = fittsMovementTime(distance, targetWidth, speed);
  const preMoveDelayMs = preMoveSettle(rng);

  const mainSampleCount = Math.max(16, Math.ceil((mt / 1000) * 120));
  const mainCurve = generateBezier(safeStart, endpoint, mainSampleCount, rng);
  const pathWithOvershoot = addOvershoot(mainCurve, endpoint, rng);
  const frameStepMs = 1000 / 60;
  const targetTimes: number[] = [];
  for (let tMs = 0; tMs < mt; tMs += frameStepMs) targetTimes.push(tMs);
  const finalInterval = mt - (targetTimes[targetTimes.length - 1] ?? 0);
  if (targetTimes.length > 1 && finalInterval < frameStepMs / 2) targetTimes[targetTimes.length - 1] = mt;
  else if (targetTimes.length === 0 || finalInterval > 1e-9) targetTimes.push(mt);
  const geometrySamples = resamplePath(pathWithOvershoot, Math.max(8, targetTimes.length * 4));
  const spatialPath: Point2D[] = [];
  for (const tMs of targetTimes) {
    const u = mt > 0 ? tMs / mt : 1;
    const warped = u * u * u * (u * (u * 6 - 15) + 10);
    const position = warped * (geometrySamples.length - 1);
    const left = Math.floor(position);
    const right = Math.min(geometrySamples.length - 1, left + 1);
    const fraction = position - left;
    const a = geometrySamples[left]!;
    const b = geometrySamples[right]!;
    spatialPath.push({ x: a.x + (b.x - a.x) * fraction, y: a.y + (b.y - a.y) * fraction });
  }
  spatialPath[0] = { x: safeStart.x, y: safeStart.y };
  spatialPath[spatialPath.length - 1] = { x: endpoint.x, y: endpoint.y };
  const temporalPath: TrajectoryPoint[] = spatialPath.map((point, i) => ({ x: point.x, y: point.y, tMs: targetTimes[i]! }));
  const tremorAmplitude = Number.isFinite(p.tremor) ? Math.max(0, p.tremor) : 0;
  const finalPath = addTremor(temporalPath, tremorAmplitude, rng);
  return { points: finalPath, durationMs: mt, preMoveDelayMs, endpoint };
}

/**
 * Convenience: synthesize a click action (movement + settle + hold).
 *
 * Returns both the trajectory and the post-arrival settle + hold times.
 */
export function synthesizeClick(
  start: Point2D,
  target: TargetBox,
  profile?: Partial<BehaviorProfile>,
  seed?: number,
): {
  trajectory: TrajectoryPlan;
  postArrivalSettleMs: number;
  clickHoldMs: number;
} {
  const rng = new RNG(seed);
  const trajectory = synthesizeMovement(start, target, profile, seed);

  // Short settle after arriving at target before clicking (Gaussian(80, 30) ms).
  const postArrivalSettleMs = Math.max(30, rng.nextGaussian(80, 30));

  // Click hold time (how long button is held down).
  const clickHoldMs = Math.max(60, rng.nextGaussian(110, 25));

  return {
    trajectory,
    postArrivalSettleMs,
    clickHoldMs,
  };
}
