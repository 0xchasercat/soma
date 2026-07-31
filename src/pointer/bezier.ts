/**
 * Cubic Bezier path generation with off-axis control points and
 * overshoot/correction submovements.
 *
 * Human intermittent-control signature: ballistic initial movement with
 * characteristic curvature, followed by corrective submovements.
 * NOT straight-line teleport or naive easing.
 *
 * Grounded in:
 *   - grimoire behav.linear_or_teleport_motion
 *   - intermittent-control-model-mouse-movements.md
 *   - mochi-behavioral-synth.md
 */

import type { Point2D } from '../types.js';
import type { RNG } from '../rng.js';

/** Cubic Bezier curve B(t) = (1−t)³P₀ + 3(1−t)²tP₁ + 3(1−t)t²P₂ + t³P₃ */
function bezierPoint(t: number, p0: Point2D, p1: Point2D, p2: Point2D, p3: Point2D): Point2D {
  const s = 1 - t;
  const a = s * s * s;
  const b = 3 * s * s * t;
  const c = 3 * s * t * t;
  const d = t * t * t;
  return {
    x: a * p0.x + b * p1.x + c * p2.x + d * p3.x,
    y: a * p0.y + b * p1.y + c * p2.y + d * p3.y,
  };
}

/**
 * Generate a human-shaped cubic Bezier curve from start to end.
 *
 * Control points are positioned off-axis to produce realistic curvature
 * (mean |Δθ|/arc_length > 0), NOT a straight line.
 *
 * @param start      Starting point.
 * @param end        Target endpoint (already sampled inside target box by fitts.ts).
 * @param sampleCount Number of uniform samples along the curve.
 * @param rng        Seeded RNG for deterministic but divergent paths.
 * @returns Array of points along the main curve (no overshoot yet).
 */
export function generateBezier(
  start: Point2D,
  end: Point2D,
  sampleCount: number,
  rng: RNG,
): Point2D[] {
  const sx = Number.isFinite(start.x) ? start.x : 0;
  const sy = Number.isFinite(start.y) ? start.y : 0;
  const ex = Number.isFinite(end.x) ? end.x : sx;
  const ey = Number.isFinite(end.y) ? end.y : sy;
  const safeStart = { x: sx, y: sy };
  const safeEnd = { x: ex, y: ey };
  const dx = ex - sx;
  const dy = ey - sy;
  const dist = Math.sqrt(dx * dx + dy * dy);
  const count = Number.isFinite(sampleCount)
    ? Math.max(2, Math.min(4096, Math.floor(sampleCount)))
    : 2;
  if (dist < 2) return [safeStart, safeEnd];

  const fwdX = dx / dist;
  const fwdY = dy / dist;
  const perpX = -fwdY;
  const perpY = fwdX;
  const p1OffsetFactor = rng.nextRange(0.3, 0.5);
  const p1PerpFactor = rng.nextRange(-0.4, 0.4);
  const p1 = {
    x: sx + fwdX * dist * p1OffsetFactor + perpX * dist * p1PerpFactor,
    y: sy + fwdY * dist * p1OffsetFactor + perpY * dist * p1PerpFactor,
  };
  const p2OffsetFactor = rng.nextRange(0.2, 0.4);
  const p2PerpFactor = rng.nextRange(-0.2, 0.2);
  const p2 = {
    x: ex - fwdX * dist * p2OffsetFactor + perpX * dist * p2PerpFactor,
    y: ey - fwdY * dist * p2OffsetFactor + perpY * dist * p2PerpFactor,
  };

  const points: Point2D[] = [];
  for (let i = 0; i < count; i++) {
    points.push(bezierPoint(i / (count - 1), safeStart, p1, p2, safeEnd));
  }
  return points;
}

/**
 * Add overshoot and correction submovement to a path.
 *
 * Human pointing: ballistic reach often overshoots the target, followed by
 * a corrective adjustment back. This produces submovement_count ≥ 1.
 *
 * @param mainPath   The initial Bezier path (ends at the target).
 * @param endpoint   The true final endpoint (where click will land).
 * @param rng        Seeded RNG.
 * @returns Extended path: mainPath → overshoot → correction → endpoint.
 */
export function addOvershoot(mainPath: Point2D[], endpoint: Point2D, rng: RNG): Point2D[] {
  if (mainPath.length < 2) return mainPath.slice();

  const finite = (v: number, fallback: number) => Number.isFinite(v) ? v : fallback;
  const requested = { x: finite(endpoint.x, mainPath[mainPath.length - 1]!.x), y: finite(endpoint.y, mainPath[mainPath.length - 1]!.y) };
  const target = mainPath[mainPath.length - 1]!;
  const origin = mainPath[0]!;
  const axisX = target.x - origin.x;
  const axisY = target.y - origin.y;
  const axisLength = Math.sqrt(axisX * axisX + axisY * axisY);
  if (axisLength < 1e-6) return [...mainPath.slice(0, -1), requested];

  // Measure beyond the target on the start-to-target axis, independent of the
  // final Bezier tangent, then correct back to the requested endpoint.
  const overshootDistance = Math.max(0, Math.min(0.15, rng.nextRange(0.05, 0.15))) * axisLength;
  const ux = axisX / axisLength;
  const uy = axisY / axisLength;
  const overshoot = { x: target.x + ux * overshootDistance, y: target.y + uy * overshootDistance };
  const correctionCount = Math.max(3, Math.min(8, Math.ceil(rng.nextRange(3, 8))));
  const correction: Point2D[] = [];
  for (let i = 1; i <= correctionCount; i++) {
    const t = i / correctionCount;
    const eased = t * t * (3 - 2 * t);
    correction.push({
      x: overshoot.x + (requested.x - overshoot.x) * eased,
      y: overshoot.y + (requested.y - overshoot.y) * eased,
    });
  }
  return [...mainPath.slice(0, -1), overshoot, ...correction];
}

/**
 * Resample a path to uniform temporal spacing.
 *
 * @param path       Input path (variable spatial density).
 * @param targetCount Desired number of samples.
 * @returns Resampled path with targetCount points.
 */
export function resamplePath(path: Point2D[], targetCount: number): Point2D[] {
  if (path.length < 2) return path;
  if (targetCount <= 1) return [path[0]!];

  // Compute cumulative arc length.
  const cumLen: number[] = [0];
  for (let i = 1; i < path.length; i++) {
    const dx = path[i]!.x - path[i - 1]!.x;
    const dy = path[i]!.y - path[i - 1]!.y;
    cumLen.push(cumLen[i - 1]! + Math.sqrt(dx * dx + dy * dy));
  }
  const totalLen = cumLen[cumLen.length - 1]!;

  // Sample at uniform arc-length intervals.
  const out: Point2D[] = [];
  for (let i = 0; i < targetCount; i++) {
    const targetLen = (i / (targetCount - 1)) * totalLen;
    // Find segment containing targetLen.
    let j = cumLen.findIndex((l) => l >= targetLen);
    if (j <= 0) j = 1;
    const segStart = cumLen[j - 1]!;
    const segEnd = cumLen[j]!;
    const t = segEnd > segStart ? (targetLen - segStart) / (segEnd - segStart) : 0;
    const p0 = path[j - 1]!;
    const p1 = path[j]!;
    out.push({
      x: p0.x + (p1.x - p0.x) * t,
      y: p0.y + (p1.y - p0.y) * t,
    });
  }
  return out;
}
