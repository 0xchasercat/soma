/**
 * Submovement count: number of corrective sub-movements (accel zero-crossings).
 *
 * Human intermittent-control: open-loop ballistic reach, then feedback-triggered
 * corrections → ≥1 acceleration reversals.
 *
 * Teleport or constant-velocity → 0 submovements.
 *
 * Grounded in:
 *   - grimoire behav.linear_or_teleport_motion (rule_in: submovement_count == 0)
 *   - intermittent-control-model-mouse-movements.md
 *   - becaptcha-mouse-synthetic-trajectories-bot-detection.md
 */
import type { TrajectoryPoint } from '../types.js';
/**
 * Count corrective submovements via acceleration zero-crossings.
 *
 * Acceleration = d(velocity)/dt. Each zero-crossing (sign change) indicates
 * a feedback-triggered correction.
 *
 * @param path  Trajectory (uniform temporal spacing recommended).
 * @returns Count of acceleration sign changes (submovements).
 */
export declare function countSubmovements(path: TrajectoryPoint[]): number;
//# sourceMappingURL=submovements.d.ts.map