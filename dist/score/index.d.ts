/** Unified scorer. Models are used only after an explicit successful load. */
import type { TrajectoryPoint, TrajectoryFeatures } from '../types.js';
export { loadJSONModel, loadJSONModelText, isJSONModelAvailable, runJSONInference, validateModelWeights, validateModelArtifact, MODEL_ARTIFACT_SCHEMA, MODEL_FEATURE_SCHEMA, } from './json_model.js';
export { extractFeatureVector, heuristicScore } from './heuristic.js';
/**
 * Score a trajectory (compute neural_bot_score).
 *
 * @param path  Trajectory to score.
 * @returns Bot probability [0, 1]. 0 = human, 1 = bot.
 */
export declare function scoreTrajectory(path: TrajectoryPoint[], _targetCenterX?: number, _targetCenterY?: number, _targetRadius?: number): Promise<number>;
/**
 * Measure trajectory and compute bot score in one call.
 *
 * @returns Full TrajectoryFeatures including neural_bot_score.
 */
export declare function measureAndScore(path: TrajectoryPoint[], _targetCenterX?: number, _targetCenterY?: number, _targetRadius?: number): Promise<TrajectoryFeatures>;
//# sourceMappingURL=index.d.ts.map