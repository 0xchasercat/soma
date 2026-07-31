/**
 * Velocity profile classification and Sigma-Lognormal fit.
 *
 * Human neuromotor model: initial acceleration → peak → final deceleration.
 * Speed envelope decomposes cleanly under Sigma-Lognormal model.
 *
 * Synthetic constant-velocity or linear easing → non-bell profile, poor fit.
 *
 * Grounded in:
 *   - grimoire behav.nonhuman_velocity_profile
 *   - becaptcha-mouse-synthetic-trajectories-bot-detection.md (Sigma-Lognormal features)
 *   - intermittent-control-model-mouse-movements.md
 */

import type { TrajectoryPoint } from '../types.js';

/**
 * Classify the velocity profile shape.
 *
 * @param speeds Array of per-frame speeds (px/s).
 * @returns 'bell' | 'linear' | 'flat' | 'unknown'.
 */
export function classifyVelocityProfile(speeds: number[]): 'bell' | 'linear' | 'flat' | 'unknown' {
  const finite = speeds.map((value) => (Number.isFinite(value) ? Math.max(0, value) : 0));
  if (finite.length < 5) return 'unknown';
  const maxSpeed = Math.max(...finite);
  if (!(maxSpeed > 1e-6)) return 'flat';
  const normalized = finite.map((value) => value / maxSpeed);
  const peakIdx = normalized.indexOf(1);
  const peakPos = peakIdx / (normalized.length - 1);
  let rises = 0;
  let falls = 0;
  let significantTurns = 0;
  let previousSign = 0;
  for (let i = 1; i < normalized.length; i++) {
    const delta = normalized[i]! - normalized[i - 1]!;
    const sign = Math.abs(delta) > 0.025 ? Math.sign(delta) : 0;
    if (sign > 0) rises++;
    if (sign < 0) falls++;
    if (sign !== 0 && previousSign !== 0 && sign !== previousSign) significantTurns++;
    if (sign !== 0) previousSign = sign;
  }
  const aboveHalf = normalized.filter((value) => value >= 0.5).length / normalized.length;
  const smoothEnough = significantTurns <= Math.max(2, Math.floor(normalized.length / 8));
  if (peakPos >= 0.2 && peakPos <= 0.8 && rises > 0 && falls > 0 && aboveHalf >= 0.15 && smoothEnough) return 'bell';
  if (rises === 0 || falls === 0) return 'linear';
  return 'unknown';
}

/**
 * Approximate a Sigma-Lognormal decomposition with up to three non-negative
 * lognormal lobes and return bounded reconstruction R².
 *
 * The input is downsampled to 64 points so capture-scale model training remains
 * deterministic and tractable without an external numerical library.
 */
export function fitSigmaLognormal(speeds: number[]): number {
  const finiteSpeeds = speeds.map((value) => (Number.isFinite(value) ? Math.max(0, value) : 0));
  if (finiteSpeeds.length < 5) return 0;
  const sampleCount = Math.min(64, finiteSpeeds.length);
  const y = Array.from({ length: sampleCount }, (_, index) => {
    const position = index * (finiteSpeeds.length - 1) / Math.max(1, sampleCount - 1);
    const left = Math.floor(position);
    const right = Math.min(finiteSpeeds.length - 1, left + 1);
    const fraction = position - left;
    return finiteSpeeds[left]! + (finiteSpeeds[right]! - finiteSpeeds[left]!) * fraction;
  });
  const maxSpeed = Math.max(...y);
  if (!(maxSpeed > 1e-6)) return 0;
  const normalized = y.map((value) => value / maxSpeed);
  const n = normalized.length;
  const peak = normalized.indexOf(1) / Math.max(1, n - 1);
  const localPeaks: number[] = [];
  for (let i = 1; i < n - 1; i++) {
    if (normalized[i]! >= normalized[i - 1]! && normalized[i]! >= normalized[i + 1]!) localPeaks.push(i / (n - 1));
  }
  localPeaks.sort((a, b) => normalized[Math.round(b * (n - 1))]! - normalized[Math.round(a * (n - 1))]!);
  const centers = [peak, ...localPeaks].filter((value, index, values) => value > 0.02 && value < 0.99 && values.indexOf(value) === index).slice(0, 3);
  if (centers.length === 0) centers.push(Math.max(0.05, Math.min(0.95, peak)));

  let bestR2 = 0;
  const candidateSets: number[][] = [centers.slice(0, 1)];
  if (centers.length >= 2) candidateSets.push(centers.slice(0, 2));
  if (centers.length >= 3) candidateSets.push(centers.slice(0, 3));
  for (const set of candidateSets) {
    for (const sigma of [0.25, 0.4, 0.65, 0.85]) {
      const basis = set.map((center) => {
        const mu = Math.log(Math.max(0.02, center)) + sigma * sigma;
        return normalized.map((_, i) => {
          const t = (i + 0.5) / n;
          const z = (Math.log(t) - mu) / sigma;
          return Math.exp(-0.5 * z * z) / (t * sigma);
        });
      });
      const coefficients = new Array(basis.length).fill(0) as number[];
      for (let pass = 0; pass < 5; pass++) {
        for (let k = 0; k < basis.length; k++) {
          let residualDot = 0;
          let basisNorm = 0;
          for (let i = 0; i < n; i++) {
            let prediction = 0;
            for (let j = 0; j < basis.length; j++) prediction += coefficients[j]! * basis[j]![i]!;
            const b = basis[k]![i]!;
            residualDot += b * (normalized[i]! - prediction + coefficients[k]! * b);
            basisNorm += b * b;
          }
          coefficients[k] = basisNorm > 1e-12 ? Math.max(0, residualDot / basisNorm) : 0;
        }
      }
      const predicted = normalized.map((_, i) => basis.reduce((sum, lobe, k) => sum + coefficients[k]! * lobe[i]!, 0));
      const predMax = Math.max(...predicted);
      if (predMax > 1e-12) bestR2 = Math.max(bestR2, computeR2(normalized, predicted.map((value) => value / predMax)));
    }
  }
  return Number.isFinite(bestR2) ? Math.max(0, Math.min(1, bestR2)) : 0;
}

/**
 * Compute R² (coefficient of determination) between observed and predicted.
 */
function computeR2(observed: number[], predicted: number[]): number {
  if (observed.length !== predicted.length || observed.length === 0) return 0;

  const mean = observed.reduce((sum, v) => sum + v, 0) / observed.length;

  let ssRes = 0;
  let ssTot = 0;

  for (let i = 0; i < observed.length; i++) {
    const obs = observed[i]!;
    const pred = predicted[i]!;
    ssRes += (obs - pred) ** 2;
    ssTot += (obs - mean) ** 2;
  }

  return ssTot > 0 ? 1 - ssRes / ssTot : 0;
}

/**
 * Extract speed profile from trajectory.
 *
 * @param path  Trajectory (uniform temporal spacing recommended).
 * @returns Array of per-frame speeds (px/s).
 */
export function extractSpeeds(path: TrajectoryPoint[]): number[] {
  if (path.length < 2) return [];
  const speeds: number[] = [];
  for (let i = 1; i < path.length; i++) {
    const p0 = path[i - 1]!;
    const p1 = path[i]!;
    const dx = p1.x - p0.x;
    const dy = p1.y - p0.y;
    const dtMs = p1.tMs - p0.tMs;
    const distance = Math.sqrt(dx * dx + dy * dy);
    speeds.push(Number.isFinite(distance) && dtMs > 0 ? distance / (dtMs / 1000) : 0);
  }
  return speeds;
}
