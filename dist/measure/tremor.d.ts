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
 * Uses a 7-point Savitzky-Golay quadratic smoother as the low-pass reference,
 * then subtracts it. A plain moving average is NOT adequate here: averaging a
 * curved path does not reproduce the curve, so on any smoothly accelerating
 * trajectory the MA residual retains the local curvature itself. Measured on a
 * zero-jitter smoothstep Bezier that leaves meanSquare ≈ 3.3 px² with 4.6% of
 * its power in the 8-12 Hz bins — i.e. a perfectly smooth synthetic curve reads
 * as having micro-tremor, which is exactly the failure mode
 * behav.missing_micro_tremor exists to catch.
 *
 * The SG quadratic kernel fits a local parabola by least squares, so constant,
 * linear AND quadratic components are removed exactly; only genuine
 * high-frequency deviation survives. Coefficients for windowSize 7, order 2:
 * (1/21)·[-2, 3, 6, 7, 6, 3, -2].
 *
 * @param path  Trajectory.
 * @returns Per-axis residuals [xResidual[], yResidual[]].
 */
export declare function detrendTrajectory(path: TrajectoryPoint[]): {
    xResidual: number[];
    yResidual: number[];
};
/**
 * Detect physiological micro-tremor in the 8-12 Hz band.
 *
 * @param residual  High-frequency residual (one axis), 60 Hz samples.
 * @returns True if band amplitude reaches the physiological floor.
 */
export declare function hasMicroTremor(residual: number[]): boolean;
/**
 * Compute lag-1 autocorrelation of a signal.
 *
 * ρ(1) = Cov(x[t], x[t−1]) / Var(x)
 *
 * Applied to the SG-quadratic-detrended residual, real hands come out NEGATIVE,
 * not positive: measured over 17,422 aimed gestures from the operator capture,
 * p50 = -0.184, p05 = -0.453, p01 = -0.552, and 84% sit below 0. Physiological
 * tremor near 10 Hz is roughly a sixth of the 60 Hz sample rate, so a residual
 * with the trend properly removed alternates frame to frame. The "ρ ≈ 0.4-0.7"
 * figure quoted for AR(1) jitter models describes the generating process at its
 * own timescale, not lag-1 of a 60 Hz high-passed pointer trace.
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