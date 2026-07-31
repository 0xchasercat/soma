/**
 * Autocorrelated micro-tremor (physiological hand jitter).
 *
 * Real human pointing: noisy biological control loop produces fine jitter
 * with autocorrelation τ ≈ 30 ms (8–12 Hz tremor band).
 *
 * WHITE NOISE (independent per-frame draws) is non-physiological even though
 * it is "noisy" — wrong autocorr structure is exactly what cside cursor_v2
 * flagged 100% in the grimoire promotion cause.
 *
 * Model: AR(1) process per axis:
 *   x[n] = ρ·x[n−1] + ε[n]
 * where:
 *   ρ = exp(−dt/τ)
 *   ε[n] ~ N(0, σ²·(1−ρ²))
 *   τ ≈ 30 ms
 *   σ = tremor_amplitude * pixelSize
 *
 * Grounded in:
 *   - grimoire behav.missing_micro_tremor
 *   - cside-cursor-v2-behavioral-detection.md
 *   - mochi-behavioral-synth.md (autocorrelated-jitter model)
 */

import type { Point2D, TrajectoryPoint } from '../types.js';
import type { RNG } from '../rng.js';

/** Autocorrelation time constant (ms). Physiological hand tremor period. */
const TAU_MS = 30;

/**
 * Add autocorrelated micro-tremor to a trajectory.
 *
 * Each axis is an independent AR(1) process with the same autocorrelation.
 *
 * @param path     Clean (smooth) trajectory points with tMs timestamps.
 * @param amplitude Tremor amplitude = profile.tremor * pixelSize.
 *                 Typical: 0.3 for moderate tremor, in CSS pixels.
 * @param rng      Seeded RNG.
 * @returns Path with physiological jitter added to each (x, y).
 */
export function addTremor(
  path: TrajectoryPoint[],
  amplitude: number,
  rng: RNG,
): TrajectoryPoint[] {
  if (path.length < 2 || amplitude <= 0) return path;

  // Initialize AR(1) state for x and y.
  let xPrev = 0;
  let yPrev = 0;

  const out: TrajectoryPoint[] = [];
  const lastIndex = path.length - 1;
  for (let i = 0; i < path.length; i++) {
    const p = path[i]!;
    if (i === 0 || i === lastIndex) {
      out.push({ x: p.x, y: p.y, tMs: p.tMs });
      continue;
    }
    const previous = path[i - 1]!;
    const rawDt = p.tMs - previous.tMs;
    const dt = Number.isFinite(rawDt) && rawDt > 0 ? rawDt : 1000 / 60;
    const rho = Math.exp(-dt / TAU_MS);
    const innovationStd = amplitude * Math.sqrt(Math.max(0, 1 - rho * rho));
    const xNoise = rho * xPrev + rng.nextGaussian(0, innovationStd);
    const yNoise = rho * yPrev + rng.nextGaussian(0, innovationStd);
    xPrev = xNoise;
    yPrev = yNoise;
    out.push({ x: p.x + xNoise, y: p.y + yNoise, tMs: p.tMs });
  }
  return out;
}

/**
 * Detrend a trajectory (remove the smooth underlying motion).
 * Returns the high-frequency residual — used by measure/tremor.ts for scoring.
 *
 * @param path  Trajectory with potential tremor.
 * @returns Per-axis residual arrays [xResidual[], yResidual[]].
 */
export function detrendTrajectory(path: TrajectoryPoint[]): {
  xResidual: number[];
  yResidual: number[];
} {
  if (path.length < 3) {
    return { xResidual: [], yResidual: [] };
  }

  // Simple moving-average low-pass filter (window = 5).
  const windowSize = 5;
  const halfWin = Math.floor(windowSize / 2);

  const xSmooth: number[] = [];
  const ySmooth: number[] = [];

  for (let i = 0; i < path.length; i++) {
    const start = Math.max(0, i - halfWin);
    const end = Math.min(path.length, i + halfWin + 1);
    let sumX = 0,
      sumY = 0,
      count = 0;
    for (let j = start; j < end; j++) {
      sumX += path[j]!.x;
      sumY += path[j]!.y;
      count++;
    }
    xSmooth.push(sumX / count);
    ySmooth.push(sumY / count);
  }

  // Residual = raw − smooth.
  const xResidual = path.map((p, i) => p.x - xSmooth[i]!);
  const yResidual = path.map((p, i) => p.y - ySmooth[i]!);

  return { xResidual, yResidual };
}
