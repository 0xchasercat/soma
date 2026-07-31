/**
 * Complete scroll sequence synthesis.
 */
import { RNG } from '../rng.js';
import { generateInertialCurve } from './inertial.js';
/**
 * Synthesize a complete inertial scroll gesture.
 *
 * @param targetDistance  Total distance to scroll in CSS pixels (positive = down, negative = up).
 * @param profile         Behavior profile (optional).
 * @param seed            RNG seed (optional — random if omitted).
 * @returns Complete scroll plan ready to dispatch (60 Hz frames).
 */
export function synthesizeScroll(targetDistance, _profile, seed) {
    const rng = new RNG(seed);
    const distance = Number.isFinite(targetDistance) ? targetDistance : 0;
    const deltas = generateInertialCurve(distance, rng);
    const frameInterval = 1000 / 60;
    const frames = deltas.map((deltaY, i) => ({ tMs: i * frameInterval, deltaY }));
    const durationMs = frames.length > 0 ? frames[frames.length - 1].tMs : 0;
    const peakDeltaY = Math.max(...deltas.map(Math.abs), 0);
    return { frames, durationMs, peakDeltaY };
}
//# sourceMappingURL=plan.js.map