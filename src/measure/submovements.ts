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
export function countSubmovements(path: TrajectoryPoint[]): number {
  if (path.length < 4) return 0;
  const speeds: number[] = [];
  for (let i = 1; i < path.length; i++) {
    const p0 = path[i - 1]!;
    const p1 = path[i]!;
    const dt = (p1.tMs - p0.tMs) / 1000;
    const dx = p1.x - p0.x;
    const dy = p1.y - p0.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    speeds.push(dt > 0 && Number.isFinite(distance) ? distance / dt : 0);
  }
  if (speeds.length < 3) return 0;

  const acceleration: number[] = [];
  for (let i = 1; i < speeds.length; i++) {
    const dt = (path[i + 1]!.tMs - path[i]!.tMs) / 1000;
    acceleration.push(dt > 0 ? (speeds[i]! - speeds[i - 1]!) / dt : 0);
  }
  const maxAcceleration = Math.max(...acceleration.map((value) => Math.abs(value)), 1);
  const threshold = Math.max(10, maxAcceleration * 0.04);
  let previousSign = 0;
  let crossings = 0;
  for (const value of acceleration) {
    const sign = Math.abs(value) >= threshold ? Math.sign(value) : 0;
    if (sign === 0) continue;
    if (previousSign !== 0 && sign !== previousSign) crossings++;
    previousSign = sign;
  }
  return Math.min(20, crossings);
}
