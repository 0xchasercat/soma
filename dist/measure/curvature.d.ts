/**
 * Curvature measurement: mean absolute turning angle per unit arc length.
 *
 * Human intermittent-control: ballistic reach with off-axis corrections → curvature > 0.
 * Straight-line teleport or naive linear easing → curvature ≈ 0.
 *
 * Grounded in:
 *   - grimoire behav.linear_or_teleport_motion (rule_in: curvature < human_min)
 *   - intermittent-control-model-mouse-movements.md
 */
import type { TrajectoryPoint } from '../types.js';
/**
 * Compute mean absolute turning angle per unit arc length (rad/px).
 *
 * For each interior sample, compute the discrete turning angle Δθ between
 * successive velocity vectors.
 *
 * curvature = Σ|Δθ| / arc_length
 *
 * @param path  Trajectory (should be resampled to uniform temporal spacing first).
 * @returns Curvature in radians per pixel. ~0 for straight line, > 0.01 for human curves.
 */
export declare function computeCurvature(path: TrajectoryPoint[]): number;
//# sourceMappingURL=curvature.d.ts.map