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
 * @param path  Trajectory.
 * @returns Per-axis residuals [xResidual[], yResidual[]].
 */
export function detrendTrajectory(path) {
    const windowSize = 7;
    if (path.length < windowSize)
        return { xResidual: [], yResidual: [] };
    const halfWin = Math.floor(windowSize / 2);
    const xResidual = [];
    const yResidual = [];
    for (let i = halfWin; i < path.length - halfWin; i++) {
        let sumX = 0;
        let sumY = 0;
        for (let j = i - halfWin; j <= i + halfWin; j++) {
            sumX += path[j].x;
            sumY += path[j].y;
        }
        xResidual.push(path[i].x - sumX / windowSize);
        yResidual.push(path[i].y - sumY / windowSize);
    }
    return { xResidual, yResidual };
}
/**
 * Detect physiological micro-tremor in the 8–12 Hz band.
 *
 * Compute energy in the tremor band via simple bandpass filter.
 *
 * @param residual  High-frequency residual (one axis).
 * @returns True if tremor energy exceeds a threshold.
 */
export function hasMicroTremor(residual) {
    if (residual.length < 12)
        return false;
    const meanSquare = residual.reduce((sum, value) => sum + value * value, 0) / residual.length;
    if (!Number.isFinite(meanSquare) || meanSquare < 1e-4)
        return false;
    return bandEnergy(residual, 60, 8, 12) > 0.012;
}
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
function bandEnergy(signal, sampleRateHz, lowHz, highHz) {
    const n = signal.length;
    if (n < 4)
        return 0;
    const mean = signal.reduce((sum, value) => sum + value, 0) / n;
    const centered = signal.map((value) => value - mean);
    const variance = centered.reduce((sum, value) => sum + value * value, 0) / n;
    if (variance < 1e-12)
        return 0;
    const firstBin = Math.max(1, Math.ceil((lowHz * n) / sampleRateHz));
    const lastBin = Math.min(Math.floor(n / 2), Math.floor((highHz * n) / sampleRateHz));
    let band = 0;
    for (let k = firstBin; k <= lastBin; k++) {
        let real = 0;
        let imaginary = 0;
        for (let i = 0; i < n; i++) {
            const phase = (2 * Math.PI * k * i) / n;
            real += centered[i] * Math.cos(phase);
            imaginary -= centered[i] * Math.sin(phase);
        }
        band += (real * real + imaginary * imaginary) / (n * n);
    }
    return Math.max(0, Math.min(1, band / (variance / 2)));
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