/**
 * Canonical trajectory feature extraction and heuristic scorer.
 *
 * The model training pipeline re-exports this extractor so training and runtime
 * always measure the same twelve inputs.
 */

import type { TrajectoryPoint, ModelFeatureVector } from '../types.js';
import { measureTrajectory } from '../measure/index.js';
import { dist2D } from '../pointer/fitts.js';

const ZERO_FEATURES: ModelFeatureVector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

function finite(value: number, fallback = 0): number {
  return Number.isFinite(value) ? value : fallback;
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, finite(value)));
}

function validPath(path: TrajectoryPoint[]): TrajectoryPoint[] {
  if (!Array.isArray(path)) return [];
  const result: TrajectoryPoint[] = [];
  let previousT = -Infinity;
  for (const point of path) {
    if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y) || !Number.isFinite(point.tMs)) continue;
    // The measurement code assumes strictly increasing timestamps.
    if (point.tMs <= previousT) continue;
    result.push(point);
    previousT = point.tMs;
  }
  return result;
}

/**
 * Extract the canonical normalized twelve-dimensional model input.
 * Invalid or degenerate trajectories produce finite zero features.
 */
export function extractFeatureVector(
  path: TrajectoryPoint[],
  _targetCenterX?: number,
  _targetCenterY?: number,
  _targetRadius?: number,
): ModelFeatureVector {
  const cleanPath = validPath(path);
  if (cleanPath.length < 2) return [...ZERO_FEATURES] as ModelFeatureVector;

  try {
    const measured = measureTrajectory(cleanPath);
    const start = cleanPath[0]!;
    const end = cleanPath[cleanPath.length - 1]!;
    const straightDist = finite(dist2D(start, end));

    let pathLength = 0;
    for (let i = 1; i < cleanPath.length; i++) {
      pathLength += finite(dist2D(cleanPath[i - 1]!, cleanPath[i]!));
    }

    const duration = Math.max(0, finite(end.tMs - start.tMs));
    // The scorer is deliberately path-only: using a fixed nominal width keeps
    // Fitts timing independent of target metadata that is absent at inference.
    const fittsEstimate = 200 + 90 * Math.log2(Math.max(straightDist, 1) / 50 + 1);

    // Directed overshoot: project every sample onto the start-to-end axis and
    // retain only excursion beyond the final endpoint.
    let maxOvershoot = 0;
    if (straightDist > 1e-9) {
      const ux = (end.x - start.x) / straightDist;
      const uy = (end.y - start.y) / straightDist;
      for (const point of cleanPath) {
        const projected = (point.x - start.x) * ux + (point.y - start.y) * uy;
        maxOvershoot = Math.max(maxOvershoot, projected - straightDist);
      }
    }
    const overshootFraction = maxOvershoot / Math.max(straightDist, 1);

    const speeds: number[] = [];
    for (let i = 1; i < cleanPath.length; i++) {
      const dt = (cleanPath[i]!.tMs - cleanPath[i - 1]!.tMs) / 1000;
      speeds.push(dt > 0 ? finite(dist2D(cleanPath[i - 1]!, cleanPath[i]!) / dt) : 0);
    }
    const maxSpeed = Math.max(...speeds, 0);
    const peakIdx = speeds.indexOf(maxSpeed);
    const speedPeakTiming = peakIdx >= 0 ? peakIdx / Math.max(1, speeds.length - 1) : 0;
    // Motion-only endpoint deceleration, normalized by peak speed.
    const finalSpeed = speeds[speeds.length - 1] ?? 0;
    const finalSpeedRatio = maxSpeed > 0 ? finalSpeed / maxSpeed : 0;
    let positiveAcceleration = 0;
    let negativeAcceleration = 0;
    for (let index = 1; index < speeds.length; index++) {
      const delta = speeds[index]! - speeds[index - 1]!;
      if (delta >= 0) positiveAcceleration += delta;
      else negativeAcceleration -= delta;
    }
    const accelerationAsymmetry = positiveAcceleration + negativeAcceleration > 0
      ? positiveAcceleration / (positiveAcceleration + negativeAcceleration)
      : 0.5;

    return [
      clamp01(measured.curvature / 0.05),
      clamp01(measured.submovement_count / 5),
      clamp01(measured.sigma_lognormal_fit),
      clamp01((finite(measured.jitter_autocorr) + 1) / 2),
      measured.has_micro_tremor ? 1 : 0,
      measured.velocity_profile === 'bell' ? 1 : 0,
      clamp01(pathLength / Math.max(straightDist, 1) - 1),
      clamp01(duration / Math.max(fittsEstimate, 1) / 2),
      clamp01(finalSpeedRatio),
      clamp01(overshootFraction / 0.3),
      clamp01(speedPeakTiming),
      clamp01(accelerationAsymmetry),
    ];
  } catch {
    return [...ZERO_FEATURES] as ModelFeatureVector;
  }
}

/** Heuristic surrogate scorer. It is not trained and is not an authenticity claim. */
export function heuristicScore(features: ModelFeatureVector): number {
  const curvatureScore = 1 - clamp01(features[0]!);
  const submovementScore = 1 - clamp01(features[1]!);
  const sigmaFitScore = 1 - clamp01(features[2]!);
  const autocorrScore = features[3]! < 0.3 ? 1 : 0;
  const tremorScore = features[4]! === 0 ? 1 : 0;
  const bellScore = features[5]! === 0 ? 1 : 0;
  return clamp01(
    curvatureScore * 0.2 +
    submovementScore * 0.15 +
    sigmaFitScore * 0.2 +
    autocorrScore * 0.2 +
    tremorScore * 0.15 +
    bellScore * 0.1,
  );
}
