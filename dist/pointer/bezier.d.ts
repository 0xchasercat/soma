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
export declare function generateBezier(start: Point2D, end: Point2D, sampleCount: number, rng: RNG): Point2D[];
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
export declare function addOvershoot(mainPath: Point2D[], endpoint: Point2D, rng: RNG): Point2D[];
/**
 * Resample a path to uniform temporal spacing.
 *
 * @param path       Input path (variable spatial density).
 * @param targetCount Desired number of samples.
 * @returns Resampled path with targetCount points.
 */
export declare function resamplePath(path: Point2D[], targetCount: number): Point2D[];
//# sourceMappingURL=bezier.d.ts.map