/**
 * Hard synthetic negatives: faithful reimplementations of the humanization
 * libraries a production detector is actually trained against.
 *
 * The easy negatives (straight line, constant velocity, white noise, naive ease,
 * random waypoints) are separable on curvature/tremor alone. They teach the model
 * nothing about the decision boundary that matters: motion that ALREADY has
 * Bezier curvature, Fitts-like timing, and jitter, yet is still machine-generated.
 *
 * Each generator below reproduces the published algorithm of a real library, so
 * the model learns what those libraries get WRONG relative to the captured human
 * corpus — which is exactly the boundary cside cursor_v2 exploits.
 *
 * Sources (algorithm shape, not vendored code):
 *   - ghost-cursor (Puppeteer): cubic Bezier through 2 random control points in the
 *     bounding box of start/end, Fitts-derived duration, fixed step count, NO tremor,
 *     NO overshoot on the common path, uniform-in-t sampling (not arc-length).
 *   - bezmouse: Bezier through randomized waypoints + per-point independent Gaussian
 *     jitter (white noise, not autocorrelated), linear-in-t timing.
 *   - HumanCursor (the Camoufox-class humanizer cside flagged 100% of the time):
 *     piecewise-linear interpolation between a few "human" waypoints with a
 *     sinusoidal ease and per-step random sleep; distortion applied to the midpoint.
 */

import type { TrajectoryPoint } from '../../src/types.js';
import { RNG } from '../../src/rng.js';

interface Point { x: number; y: number }

/** Cubic Bezier evaluation, shared by the Bezier-family generators. */
function cubic(t: number, p0: Point, p1: Point, p2: Point, p3: Point): Point {
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
 * ghost-cursor style path.
 *
 * Defect vs human: control points are sampled anywhere in the start/end bounding
 * box, so curvature is uncontrolled (often near-zero for axis-aligned moves);
 * sampling is uniform in t rather than arc length, so the speed profile is a
 * Bezier artifact rather than a neuromotor bell; there is no physiological tremor
 * and no corrective submovement, so submovement_count collapses to 0.
 */
export function generateGhostCursor(start: Point, end: Point, rng: RNG): TrajectoryPoint[] {
  // ghost-cursor: Fitts-ish duration from distance, clamped.
  const distance = Math.hypot(end.x - start.x, end.y - start.y);
  const durationMs = Math.min(2000, Math.max(100, 100 + distance * rng.nextRange(1.2, 2.4)));

  // Control points uniformly inside the bounding box spanned by start/end.
  const minX = Math.min(start.x, end.x);
  const maxX = Math.max(start.x, end.x);
  const minY = Math.min(start.y, end.y);
  const maxY = Math.max(start.y, end.y);
  const p1 = { x: rng.nextRange(minX, maxX), y: rng.nextRange(minY, maxY) };
  const p2 = { x: rng.nextRange(minX, maxX), y: rng.nextRange(minY, maxY) };

  // Fixed step count, uniform in t — no arc-length resampling.
  const steps = Math.max(12, Math.min(60, Math.round(distance / 8)));
  const path: TrajectoryPoint[] = [];
  for (let index = 0; index <= steps; index++) {
    const t = index / steps;
    const point = cubic(t, start, p1, p2, end);
    path.push({ x: point.x, y: point.y, tMs: t * durationMs });
  }
  return path;
}

/**
 * bezmouse style path.
 *
 * Defect vs human: jitter is drawn independently per sample, so the residual is
 * white noise — jitter_autocorr sits near 0 while a real hand autocorrelates at
 * tau ~= 30 ms. Timing is linear in t, so there is no accel/decel envelope.
 */
export function generateBezmouse(start: Point, end: Point, rng: RNG): TrajectoryPoint[] {
  const distance = Math.hypot(end.x - start.x, end.y - start.y);
  const durationMs = Math.max(120, distance * rng.nextRange(1.5, 3.0));

  // Randomized interior waypoints, then a Bezier through them (bezmouse deviates
  // perpendicular to the axis by a random amount).
  const deviation = rng.nextRange(5, 40);
  const midX = (start.x + end.x) / 2;
  const midY = (start.y + end.y) / 2;
  const p1 = { x: midX + rng.nextRange(-deviation, deviation), y: midY + rng.nextRange(-deviation, deviation) };
  const p2 = { x: midX + rng.nextRange(-deviation, deviation), y: midY + rng.nextRange(-deviation, deviation) };

  const steps = Math.max(15, Math.min(80, Math.round(distance / 6)));
  const path: TrajectoryPoint[] = [];
  for (let index = 0; index <= steps; index++) {
    const t = index / steps;
    const point = cubic(t, start, p1, p2, end);
    // Independent per-sample Gaussian jitter: the white-noise tell.
    path.push({
      x: point.x + rng.nextGaussian(0, 1.2),
      y: point.y + rng.nextGaussian(0, 1.2),
      tMs: t * durationMs,
    });
  }
  return path;
}

/**
 * HumanCursor style path (the Camoufox-class humanizer).
 *
 * Defect vs human: the path is piecewise-linear between a handful of waypoints
 * with a sinusoidal ease per segment, so curvature concentrates at the waypoint
 * joins instead of distributing along a smooth reach, and the velocity profile is
 * multi-peaked rather than a single dominant lognormal lobe. No tremor.
 */
export function generateHumanCursor(start: Point, end: Point, rng: RNG): TrajectoryPoint[] {
  const distance = Math.hypot(end.x - start.x, end.y - start.y);
  const durationMs = Math.max(150, distance * rng.nextRange(1.8, 3.5));

  // A few interior waypoints with midpoint "distortion".
  const waypointCount = 2 + rng.nextInt(3);
  const waypoints: Point[] = [start];
  for (let index = 1; index <= waypointCount; index++) {
    const t = index / (waypointCount + 1);
    const spread = distance * rng.nextRange(0.02, 0.09);
    waypoints.push({
      x: start.x + (end.x - start.x) * t + rng.nextRange(-spread, spread),
      y: start.y + (end.y - start.y) * t + rng.nextRange(-spread, spread),
    });
  }
  waypoints.push(end);

  // Linear interpolation per segment under a sinusoidal ease.
  const stepsPerSegment = Math.max(4, Math.round(60 / waypoints.length));
  const path: TrajectoryPoint[] = [];
  const totalSteps = (waypoints.length - 1) * stepsPerSegment;
  let emitted = 0;
  for (let segment = 0; segment < waypoints.length - 1; segment++) {
    const from = waypoints[segment]!;
    const to = waypoints[segment + 1]!;
    for (let step = segment === 0 ? 0 : 1; step <= stepsPerSegment; step++) {
      const local = step / stepsPerSegment;
      const eased = 0.5 - 0.5 * Math.cos(Math.PI * local);
      path.push({
        x: from.x + (to.x - from.x) * eased,
        y: from.y + (to.y - from.y) * eased,
        tMs: (emitted / totalSteps) * durationMs,
      });
      emitted++;
    }
  }
  return path;
}

/** Every hard-negative generator, addressed by a stable source label. */
export const HARD_NEGATIVE_GENERATORS: ReadonlyArray<{
  source: string;
  generate: (start: Point, end: Point, rng: RNG) => TrajectoryPoint[];
}> = [
  { source: 'ghost_cursor', generate: generateGhostCursor },
  { source: 'bezmouse', generate: generateBezmouse },
  { source: 'human_cursor', generate: generateHumanCursor },
];
