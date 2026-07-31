/**
 * Canonical trajectory feature extraction and heuristic scorer.
 *
 * The model training pipeline re-exports this extractor so training and runtime
 * always measure the same twelve inputs.
 */
import type { TrajectoryPoint, ModelFeatureVector } from '../types.js';
/**
 * Extract the canonical normalized twelve-dimensional model input.
 * Invalid or degenerate trajectories produce finite zero features.
 */
export declare function extractFeatureVector(path: TrajectoryPoint[], _targetCenterX?: number, _targetCenterY?: number, _targetRadius?: number): ModelFeatureVector;
/** Heuristic surrogate scorer. It is not trained and is not an authenticity claim. */
export declare function heuristicScore(features: ModelFeatureVector): number;
//# sourceMappingURL=heuristic.d.ts.map