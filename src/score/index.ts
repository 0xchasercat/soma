/** Unified scorer. Models are used only after an explicit successful load. */

import type { TrajectoryPoint, TrajectoryFeatures } from '../types.js';
import { measureTrajectory } from '../measure/index.js';
import { extractFeatureVector, heuristicScore } from './heuristic.js';
import { runJSONInference, isJSONModelAvailable } from './json_model.js';
import { runONNXInference, isONNXAvailable } from './onnx.js';

export {
  loadJSONModel,
  loadJSONModelText,
  isJSONModelAvailable,
  runJSONInference,
  validateModelWeights,
  validateModelArtifact,
  MODEL_ARTIFACT_SCHEMA,
  MODEL_FEATURE_SCHEMA,
} from './json_model.js';
export { extractFeatureVector, heuristicScore } from './heuristic.js';

/**
 * Score a trajectory (compute neural_bot_score).
 *
 * @param path  Trajectory to score.
 * @returns Bot probability [0, 1]. 0 = human, 1 = bot.
 */
export async function scoreTrajectory(
  path: TrajectoryPoint[],
  _targetCenterX?: number,
  _targetCenterY?: number,
  _targetRadius?: number,
): Promise<number> {
  const features = extractFeatureVector(path);

  // JSON model: trained weights, no native dep.
  if (isJSONModelAvailable()) {
    const jsonScore = runJSONInference(features);
    if (jsonScore !== null) return jsonScore;
  }

  // ONNX model: optional native runtime, also loaded explicitly.
  if (isONNXAvailable()) {
    const onnxScore = await runONNXInference(features);
    if (onnxScore !== null) return onnxScore;
  }

  // No valid model is loaded: use the documented heuristic fallback.
  return heuristicScore(features);
}

/**
 * Measure trajectory and compute bot score in one call.
 *
 * @returns Full TrajectoryFeatures including neural_bot_score.
 */
export async function measureAndScore(
  path: TrajectoryPoint[],
  _targetCenterX?: number,
  _targetCenterY?: number,
  _targetRadius?: number,
): Promise<TrajectoryFeatures> {
  const measured = measureTrajectory(path);
  const neural_bot_score = await scoreTrajectory(path);

  return {
    ...measured,
    neural_bot_score,
  };
}
