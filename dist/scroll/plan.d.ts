/**
 * Complete scroll sequence synthesis.
 */
import type { ScrollPlan, BehaviorProfile } from '../types.js';
/**
 * Synthesize a complete inertial scroll gesture.
 *
 * @param targetDistance  Total distance to scroll in CSS pixels (positive = down, negative = up).
 * @param profile         Behavior profile (optional).
 * @param seed            RNG seed (optional — random if omitted).
 * @returns Complete scroll plan ready to dispatch (60 Hz frames).
 */
export declare function synthesizeScroll(targetDistance: number, _profile?: Partial<BehaviorProfile>, seed?: number): ScrollPlan;
//# sourceMappingURL=plan.d.ts.map