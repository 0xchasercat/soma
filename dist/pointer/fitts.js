/**
 * Fitts's Law for human pointing movement time estimation.
 *
 * MT = a + b · log₂(D/W + 1)
 *
 * Constants from mochi's documented model:
 *   a = 200 ms  (perceptual-motor reaction intercept)
 *   b = 90 ms/bit
 *
 * These are the values the grimoire differential uses as the human model:
 *   behav.nonhuman_velocity_profile → shape the path so Sigma-Lognormal
 *   decomposition yields init-accel + final-decel consistent with Fitts MT.
 *   behav.superhuman_speed → action latency must exceed the Fitts intercept.
 */
const FITTS_A_MS = 200; // reaction-time intercept (ms)
const FITTS_B_MS = 90; // throughput constant (ms/bit)
/**
 * Predicted movement time for a pointing action.
 *
 * @param distPx       Euclidean distance from start to target centre (px).
 * @param targetWidthPx  Width of the target in the movement direction (px).
 * @param speedMult    From BehaviorProfile.speed — multiplies 1/MT not MT,
 *                     so speed > 1 reduces duration (faster), < 1 increases it.
 * @returns Movement time in ms. Never below 80 ms (physical minimum).
 */
export function fittsMovementTime(distPx, targetWidthPx, speedMult = 1.0) {
    // Avoid log(0) — clamp minimum distance and width.
    const d = Math.max(distPx, 1);
    const w = Math.max(targetWidthPx, 4);
    const id = Math.log2(d / w + 1); // index of difficulty (bits)
    const mt = FITTS_A_MS + FITTS_B_MS * id;
    return Math.max(80, mt / speedMult);
}
/**
 * Euclidean distance between two points.
 */
export function dist2D(a, b) {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    return Math.sqrt(dx * dx + dy * dy);
}
/**
 * Sample a realistic target endpoint inside the given bounding box.
 *
 * Human pointing does not always land exactly at the visual centre —
 * it is biased toward centre but with spread controlled by precision.
 *
 * @param box       Target bounding box in viewport coordinates.
 * @param precision BehaviorProfile.precision ∈ [0, 1].
 *                  1 = Gaussian strongly centred; 0 = uniform in box.
 * @param rng       Seeded RNG.
 * @returns Endpoint in viewport coordinates.
 */
export function sampleEndpoint(box, precision, rng) {
    const left = Math.min(box.x, box.x + box.width);
    const right = Math.max(box.x, box.x + box.width);
    const top = Math.min(box.y, box.y + box.height);
    const bottom = Math.max(box.y, box.y + box.height);
    const width = right - left;
    const height = bottom - top;
    const cx = left + width / 2;
    const cy = top + height / 2;
    if (precision >= 0.999 || width === 0 || height === 0)
        return { x: cx, y: cy };
    const sigX = (width / 2) * (1 - precision) * 0.6;
    const sigY = (height / 2) * (1 - precision) * 0.6;
    const insetX = Math.min(2, width / 2);
    const insetY = Math.min(2, height / 2);
    const x = Math.max(left + insetX, Math.min(right - insetX, rng.nextGaussian(cx, sigX)));
    const y = Math.max(top + insetY, Math.min(bottom - insetY, rng.nextGaussian(cy, sigY)));
    return { x, y };
}
/**
 * Pre-move settle delay: the perceptual-motor preparation interval
 * before the pointer physically begins moving. Drawn from Gaussian(150, 50) ms.
 *
 * This is the Fitts intercept a=200 ms in spirit; the 150 ms here is the
 * pre-movement cognitive component (the remaining ~50 ms is absorbed into
 * the early acceleration phase of the trajectory itself).
 */
export function preMoveSettle(rng) {
    return Math.max(40, rng.nextGaussian(150, 50));
}
/**
 * Post-click dwell before releasing the button (click hold time).
 * Human button presses are not instantaneous — typically 80–180 ms.
 */
export function clickHoldTime(rng) {
    return Math.max(60, rng.nextGaussian(110, 25));
}
//# sourceMappingURL=fitts.js.map