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
export function detrendTrajectory(path) {
    const windowSize = 7;
    if (path.length < windowSize)
        return { xResidual: [], yResidual: [] };
    const kernel = [-2, 3, 6, 7, 6, 3, -2];
    const norm = 21;
    const halfWin = Math.floor(windowSize / 2);
    const xResidual = [];
    const yResidual = [];
    for (let i = halfWin; i < path.length - halfWin; i++) {
        let sumX = 0;
        let sumY = 0;
        for (let k = 0; k < windowSize; k++) {
            const p = path[i - halfWin + k];
            sumX += kernel[k] * p.x;
            sumY += kernel[k] * p.y;
        }
        xResidual.push(path[i].x - sumX / norm);
        yResidual.push(path[i].y - sumY / norm);
    }
    return { xResidual, yResidual };
}
/**
 * RMS amplitude (px) of a residual within [lowHz, highHz], via Parseval.
 *
 * Absolute, rather than the band's share of the residual's own variance. A
 * relative measure cannot distinguish tremor from detrend leakage: a
 * zero-jitter curve has a tiny residual whose leaked power
 * still lands in the tremor bins, so its RATIO passes while its AMPLITUDE is
 * ~100x below a real hand's (measured smooth p99 = 0.063 px vs real p01 =
 * 0.033 px, real p50 = 0.360 px).
 */
function bandRmsPx(signal, sampleRateHz, lowHz, highHz) {
    const n = signal.length;
    if (n < 4)
        return 0;
    let mean = 0;
    for (let i = 0; i < n; i++)
        mean += signal[i];
    mean /= n;
    const firstBin = Math.max(1, Math.ceil((lowHz * n) / sampleRateHz));
    const lastBin = Math.min(Math.floor(n / 2), Math.floor((highHz * n) / sampleRateHz));
    if (lastBin < firstBin)
        return 0;
    let power = 0;
    for (let k = firstBin; k <= lastBin; k++) {
        let real = 0;
        let imaginary = 0;
        for (let i = 0; i < n; i++) {
            const phase = (2 * Math.PI * k * i) / n;
            const centered = signal[i] - mean;
            real += centered * Math.cos(phase);
            imaginary -= centered * Math.sin(phase);
        }
        power += (2 * (real * real + imaginary * imaginary)) / (n * n);
    }
    return Number.isFinite(power) && power > 0 ? Math.sqrt(power) : 0;
}
/**
 * Minimum physiological tremor amplitude in the 8-12 Hz band, in pixels.
 *
 * Set at the real-human p01 over 17,422 aimed gestures from the operator
 * capture (p01 = 0.0327 px, p50 = 0.3602 px), i.e. a 1% false-negative budget
 * on genuine motion — the same convention as the sip-eval thresholds. A
 * zero-jitter analytic curve reaches this in 10% of cases (its p99 is
 * 0.0627 px); those residual smooth paths are caught by the neural motion
 * scorer, which fires on all of them.
 */
const TREMOR_MIN_BAND_RMS_PX = 0.03;
/**
 * Detect physiological micro-tremor in the 8-12 Hz band.
 *
 * @param residual  High-frequency residual (one axis), 60 Hz samples.
 * @returns True if band amplitude reaches the physiological floor.
 */
export function hasMicroTremor(residual) {
    if (residual.length < 12)
        return false;
    return bandRmsPx(residual, 60, 8, 12) > TREMOR_MIN_BAND_RMS_PX;
}
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
export function computeAutocorrelation(signal) {
    if (signal.length < 3)
        return 0;
    const finite = signal.map((value) => Number.isFinite(value) ? value : 0);
    const mean = finite.reduce((sum, value) => sum + value, 0) / finite.length;
    let variance = 0;
    let covariance = 0;
    for (let i = 0; i < finite.length; i++) {
        const centered = finite[i] - mean;
        variance += centered * centered;
        if (i > 0)
            covariance += centered * (finite[i - 1] - mean);
    }
    if (variance < 1e-12)
        return 0;
    return Math.max(-1, Math.min(1, covariance / variance));
}
/**
 * Measure micro-tremor presence and autocorrelation for a trajectory.
 *
 * @param path  Trajectory (uniform temporal spacing recommended).
 * @returns { has_micro_tremor, jitter_autocorr }.
 */
export function measureTremor(path) {
    const { xResidual, yResidual } = detrendTrajectory(path);
    const has_micro_tremor = hasMicroTremor(xResidual) || hasMicroTremor(yResidual);
    const autocorrX = computeAutocorrelation(xResidual);
    const autocorrY = computeAutocorrelation(yResidual);
    const jitter_autocorr = Number.isFinite(autocorrX + autocorrY) ? (autocorrX + autocorrY) / 2 : 0;
    return { has_micro_tremor, jitter_autocorr };
}
//# sourceMappingURL=tremor.js.map