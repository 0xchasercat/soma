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
/**
 * AR(2) coefficients for the tremor oscillator, fitted through `detrendTrajectory`
 * against the measured real residual ACF. Defined at the 60 Hz frame rate the
 * synthesizer emits. Damping 0.98, frequency 4.0 Hz.
 */
const AR2_PHI1 = 1.7905;
const AR2_PHI2 = -0.9604;
/** Reference frame interval (ms) the coefficients were fitted at. */
const FRAME_MS = 1000 / 60;
/**
 * Stationary variance of the AR(2) process driven by unit-variance innovations.
 *
 * var = (1 - phi2) / ((1 + phi2) * (1 - phi1 - phi2) * (1 + phi1 - phi2))
 *
 * The innovation is divided by the square root of this so the emitted residual has the
 * requested amplitude regardless of the coefficients.
 */
const AR2_STATIONARY_VARIANCE = (1 - AR2_PHI2) /
    ((1 + AR2_PHI2) * (1 - AR2_PHI1 - AR2_PHI2) * (1 + AR2_PHI1 - AR2_PHI2));
const AR2_INNOVATION_SCALE = 1 / Math.sqrt(AR2_STATIONARY_VARIANCE);
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
export function addTremor(path, amplitude, rng) {
    if (path.length < 2 || amplitude <= 0)
        return path;
    // AR(2) needs two lags of history per axis. Starting from zero would be wrong here:
    // damping is 0.98, so the transient needs ~160 samples to reach stationarity while a
    // typical reach is ~40. Starting cold, the whole gesture stays inside the ramp and the
    // emitted amplitude is governed by transient length rather than by profile.tremor —
    // measured, a 0.25-to-0.45 coefficient sweep moved emitted residual std by 0.017px,
    // i.e. the persona's tremor value was effectively ignored. Seeding both lags from the
    // stationary distribution starts the oscillator already running.
    const stationaryStd = amplitude;
    let xPrev1 = rng.nextGaussian(0, stationaryStd);
    let xPrev2 = rng.nextGaussian(0, stationaryStd);
    let yPrev1 = rng.nextGaussian(0, stationaryStd);
    let yPrev2 = rng.nextGaussian(0, stationaryStd);
    const out = [];
    const lastIndex = path.length - 1;
    for (let i = 0; i < path.length; i++) {
        const p = path[i];
        if (i === 0 || i === lastIndex) {
            // Endpoints stay exact: the path must start at the cursor and land on target.
            out.push({ x: p.x, y: p.y, tMs: p.tMs });
            continue;
        }
        const previous = path[i - 1];
        const rawDt = p.tMs - previous.tMs;
        const dt = Number.isFinite(rawDt) && rawDt > 0 ? rawDt : FRAME_MS;
        // The coefficients are fitted at 60 Hz. When frames arrive at a different rate the
        // oscillator would advance at the wrong phase, so scale the innovation by the
        // interval ratio rather than silently mis-stepping.
        const rateScale = Math.sqrt(Math.min(4, Math.max(0.25, dt / FRAME_MS)));
        const innovationStd = amplitude * AR2_INNOVATION_SCALE * rateScale;
        const xNoise = AR2_PHI1 * xPrev1 + AR2_PHI2 * xPrev2 + rng.nextGaussian(0, innovationStd);
        const yNoise = AR2_PHI1 * yPrev1 + AR2_PHI2 * yPrev2 + rng.nextGaussian(0, innovationStd);
        xPrev2 = xPrev1;
        xPrev1 = xNoise;
        yPrev2 = yPrev1;
        yPrev1 = yNoise;
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
export function detrendTrajectory(path) {
    if (path.length < 3) {
        return { xResidual: [], yResidual: [] };
    }
    // Simple moving-average low-pass filter (window = 5).
    const windowSize = 5;
    const halfWin = Math.floor(windowSize / 2);
    const xSmooth = [];
    const ySmooth = [];
    for (let i = 0; i < path.length; i++) {
        const start = Math.max(0, i - halfWin);
        const end = Math.min(path.length, i + halfWin + 1);
        let sumX = 0, sumY = 0, count = 0;
        for (let j = start; j < end; j++) {
            sumX += path[j].x;
            sumY += path[j].y;
            count++;
        }
        xSmooth.push(sumX / count);
        ySmooth.push(sumY / count);
    }
    // Residual = raw − smooth.
    const xResidual = path.map((p, i) => p.x - xSmooth[i]);
    const yResidual = path.map((p, i) => p.y - ySmooth[i]);
    return { xResidual, yResidual };
}
//# sourceMappingURL=tremor.js.map