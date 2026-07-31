/**
 * Micro-tremor detection and autocorrelation measurement.
 *
 * Real hand: physiological jitter in 8–12 Hz tremor band, autocorrelated (τ≈30ms).
 * Synthetic: either too smooth OR white-noise jitter (autocorr ≈ 0) — both non-physiological.
 *
 * Wrong autocorr structure is exactly what cside cursor_v2 exploited to flag
 * a motion-humanized agent 100% despite flawless fingerprint stealth.
 *
 * Grounded in:
 *   - grimoire behav.missing_micro_tremor
 *   - cside-cursor-v2-behavioral-detection.md
 *   - mochi-behavioral-synth.md (autocorrelated jitter τ≈30ms)
 */
import type { TrajectoryPoint } from '../types.js';
/**
 * Detrend trajectory and extract high-frequency residual.
 *
 * @param path  Trajectory.
 * @returns Per-axis residuals [xResidual[], yResidual[]].
 */
export declare function detrendTrajectory(path: TrajectoryPoint[]): {
    xResidual: number[];
    yResidual: number[];
};
/**
 * Detect physiological micro-tremor in the 8–12 Hz band.
 *
 * Compute energy in the tremor band via simple bandpass filter.
 *
 * @param residual  High-frequency residual (one axis).
 * @returns True if tremor energy exceeds a threshold.
 */
export declare function hasMicroTremor(residual: number[]): boolean;
/**
 * Compute lag-1 autocorrelation of a signal.
 *
 * ρ(1) = Cov(x[t], x[t−1]) / Var(x)
 *
 * Human AR(1) jitter: ρ ≈ 0.4–0.7 (τ≈30ms at 60Hz).
 * White noise: ρ ≈ 0.
 * Perfect smooth (no jitter): ρ ≈ 0 (or undefined if variance = 0).
 *
 * @param signal  Time series (e.g. residual per axis).
 * @returns Lag-1 autocorrelation ∈ [−1, 1].
 */
export declare function computeAutocorrelation(signal: number[]): number;
/**
 * Measure micro-tremor presence and autocorrelation for a trajectory.
 *
 * @param path  Trajectory (uniform temporal spacing recommended).
 * @returns { has_micro_tremor, jitter_autocorr }.
 */
export declare function measureTremor(path: TrajectoryPoint[]): {
    has_micro_tremor: boolean;
    jitter_autocorr: number;
};
//# sourceMappingURL=tremor.d.ts.map