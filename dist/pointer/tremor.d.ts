/**
 * Physiological micro-tremor as a damped oscillator.
 *
 * A real hand is not a smoothed random walk — it is a mechanical control loop that
 * oscillates. Measured over 9,040 real reaches (>=150px, >=20 resampled samples), the
 * detrended residual autocorrelation is:
 *
 *   lag   1      2      3      4      5      6      7      8      9     10
 *   acf   0.647  0.244  0.009 -0.149 -0.220 -0.223 -0.192 -0.149 -0.109 -0.076
 *
 * It crosses zero at lag ~3.06 and troughs at −0.223 around lag 6. An AR(1) process
 * CANNOT produce that: its ACF is rho^k, which decays monotonically and never changes
 * sign. The previous AR(1) model was therefore structurally incapable of matching real
 * tremor, not merely mistuned — sweeping tau from 30ms to 160ms moved lag-1 correlation
 * from 0.29 to 0.56 while leaving the sign reversal unreachable, and left synthesized
 * curvature ~5x real on short reaches.
 *
 * Zero crossing at lag 3 and trough at lag 6 put the half-period near 100ms, i.e. a
 * period near 200ms — a genuine oscillation, so the process needs complex conjugate
 * roots. An AR(2) supplies them: x[n] = phi1*x[n-1] + phi2*x[n-2] + eps[n], oscillatory
 * when phi1^2 + 4*phi2 < 0.
 *
 * The coefficients are fitted in MEASURED space, not analytically. A textbook
 * Yule-Walker fit on lags 1-2 (phi1 0.8413, phi2 -0.3003) reproduces the sign change
 * but damps far too fast, predicting a lag-6 trough of -0.029 against the real -0.223.
 * Worse, the estimator is not transparent: `detrendTrajectory` subtracts a smooth fit,
 * which reshapes the autocorrelation it reports, so coefficients that are correct in
 * theory measure wrong in practice. Sweeping the oscillator frequency from 4.4Hz to
 * 10.5Hz moved the end-to-end measured ACF by under 0.002 MAE — that parameter was not
 * controlling the observable.
 *
 * So the fit is done through the same estimator the scorer uses: grid search over
 * (damping, frequency), synthesizing the AR(2), passing it through
 * `detrendTrajectory`, and comparing the reported ACF against the real curve. Best fit
 * is damping 0.98 at 4.0Hz (MAE 0.051 across lags 1-7), giving the coefficients below.
 * That reproduces the zero crossing at lag 3 and a lag-6 trough of -0.255 against the
 * real -0.223.
 *
 * Grounded in:
 *   - grimoire behav.missing_micro_tremor (jitter_autocorr is a rule_in signal)
 *   - cside-cursor-v2-behavioral-detection.md (wrong noise STRUCTURE is the tell,
 *     independent of noise magnitude)
 *   - operator capture 2026-08-04, residual ACF over 9,040 reaches
 */
import type { TrajectoryPoint } from '../types.js';
import type { RNG } from '../rng.js';
/**
 * Add physiological micro-tremor to a trajectory.
 *
 * Each axis runs an independent AR(2) damped oscillator. Axes share coefficients but
 * not state, so they oscillate incoherently the way two degrees of freedom of one hand
 * do.
 *
 * @param path      Clean (smooth) trajectory points with tMs timestamps.
 * @param amplitude Target residual standard deviation in CSS pixels
 *                  (profile.tremor; DEFAULT_PROFILE uses 0.3).
 * @param rng       Seeded RNG.
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