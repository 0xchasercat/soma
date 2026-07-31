/**
 * Complete pointer trajectory synthesis.
 *
 * Composes: Fitts timing → Bezier path → overshoot → tremor → 60 Hz resampling.
 *
 * The output TrajectoryPlan is the complete synthesis result ready to dispatch
 * via CDP Input.dispatchMouseEvent or equivalent trusted-event channel.
 */
import type { Point2D, TargetBox, BehaviorProfile, TrajectoryPlan } from '../types.js';
/**
 * Synthesize a complete pointer movement from start to target.
 *
 * @param start   Starting cursor position (viewport coordinates).
 * @param target  Target bounding box (already resolved via DOM.getBoxModel or equivalent).
 * @param profile Behavior profile (optional — uses DEFAULT_PROFILE if omitted).
 * @param seed    RNG seed (optional — random if omitted).
 * @returns Complete trajectory plan ready to dispatch.
 */
export declare function synthesizeMovement(start: Point2D, target: TargetBox, profile?: Partial<BehaviorProfile>, seed?: number): TrajectoryPlan;
/**
 * Convenience: synthesize a click action (movement + settle + hold).
 *
 * Returns both the trajectory and the post-arrival settle + hold times.
 */
export declare function synthesizeClick(start: Point2D, target: TargetBox, profile?: Partial<BehaviorProfile>, seed?: number): {
    trajectory: TrajectoryPlan;
    postArrivalSettleMs: number;
    clickHoldMs: number;
};
//# sourceMappingURL=trajectory.d.ts.map