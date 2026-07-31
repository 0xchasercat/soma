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
import type { RNG } from '../rng.js';
import type { Point2D, TargetBox } from '../types.js';
/**
 * Predicted movement time for a pointing action.
 *
 * @param distPx       Euclidean distance from start to target centre (px).
 * @param targetWidthPx  Width of the target in the movement direction (px).
 * @param speedMult    From BehaviorProfile.speed — multiplies 1/MT not MT,
 *                     so speed > 1 reduces duration (faster), < 1 increases it.
 * @returns Movement time in ms. Never below 80 ms (physical minimum).
 */
export declare function fittsMovementTime(distPx: number, targetWidthPx: number, speedMult?: number): number;
/**
 * Euclidean distance between two points.
 */
export declare function dist2D(a: Point2D, b: Point2D): number;
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
export declare function sampleEndpoint(box: TargetBox, precision: number, rng: RNG): Point2D;
/**
 * Pre-move settle delay: the perceptual-motor preparation interval
 * before the pointer physically begins moving. Drawn from Gaussian(150, 50) ms.
 *
 * This is the Fitts intercept a=200 ms in spirit; the 150 ms here is the
 * pre-movement cognitive component (the remaining ~50 ms is absorbed into
 * the early acceleration phase of the trajectory itself).
 */
export declare function preMoveSettle(rng: RNG): number;
/**
 * Post-click dwell before releasing the button (click hold time).
 * Human button presses are not instantaneous — typically 80–180 ms.
 */
export declare function clickHoldTime(rng: RNG): number;
//# sourceMappingURL=fitts.d.ts.map