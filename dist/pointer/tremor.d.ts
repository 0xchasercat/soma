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
import type { TrajectoryPoint } from '../types.js';
import type { RNG } from '../rng.js';
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
export declare function addTremor(path: TrajectoryPoint[], amplitude: number, rng: RNG): TrajectoryPoint[];
/**
 * Detrend a trajectory (remove the smooth underlying motion).
 * Returns the high-frequency residual — used by measure/tremor.ts for scoring.
 *
 * @param path  Trajectory with potential tremor.
 * @returns Per-axis residual arrays [xResidual[], yResidual[]].
 */
export declare function detrendTrajectory(path: TrajectoryPoint[]): {
    xResidual: number[];
    yResidual: number[];
};
//# sourceMappingURL=tremor.d.ts.map