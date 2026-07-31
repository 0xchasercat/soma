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
export function computeCurvature(path: TrajectoryPoint[]): number {
  if (path.length < 3) return 0;
  let totalTurning = 0;
  let totalArcLength = 0;
  for (let i = 1; i < path.length - 1; i++) {
    const p0 = path[i - 1]!;
    const p1 = path[i]!;
    const p2 = path[i + 1]!;
    const v1x = p1.x - p0.x;
    const v1y = p1.y - p0.y;
    const v2x = p2.x - p1.x;
    const v2y = p2.y - p1.y;
    const mag1 = Math.sqrt(v1x * v1x + v1y * v1y);
    const mag2 = Math.sqrt(v2x * v2x + v2y * v2y);
    if (!(mag1 > 1e-6) || !(mag2 > 1e-6)) continue;
    const dot = (v1x * v2x + v1y * v2y) / (mag1 * mag2);
    totalTurning += Math.acos(Math.max(-1, Math.min(1, dot)));
    totalArcLength += mag1;
  }
  const last = path[path.length - 1]!;
  const previous = path[path.length - 2]!;
  totalArcLength += Math.hypot(last.x - previous.x, last.y - previous.y);
  const result = totalArcLength > 1e-9 ? totalTurning / totalArcLength : 0;
  return Number.isFinite(result) ? result : 0;
}
