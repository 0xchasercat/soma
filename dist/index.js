/**
 * Soma — behavioral synthesis and measurement primitives grounded in the
 * behavioral contracts documented by clandestine-grimoire.
 *
 * The library provides deterministic plans and local measurement/scoring. It
 * does not claim that generated input passes any third-party detector.
 */
// ─── RNG & Profile ───────────────────────────────────────────────────────────
export { RNG, defaultRNG } from './rng.js';
export { DEFAULT_PROFILE, PROFILES, mergeProfile, clampProfile } from './profile.js';
// ─── Synthesis ───────────────────────────────────────────────────────────────
export { synthesizeMovement, synthesizeClick } from './pointer/trajectory.js';
export { synthesizeKeystrokes } from './keystroke/plan.js';
export { synthesizeScroll } from './scroll/plan.js';
export { generateDwell, generatePostActionSettle, generateInterActionInterval } from './cadence/timing.js';
// ─── Measurement ─────────────────────────────────────────────────────────────
export { measureTrajectory, resample60Hz, computeCurvature, countSubmovements, extractSpeeds, classifyVelocityProfile, fitSigmaLognormal, measureTremor, extractKeystrokeFeatures, extractCapturedKeystrokeFeatures, extractCadenceFeatures, extractCapturedInteractionFeatures, } from './measure/index.js';
// ─── Scoring ─────────────────────────────────────────────────────────────────
export { scoreTrajectory, measureAndScore, loadJSONModel, loadJSONModelText, validateModelWeights, validateModelArtifact, MODEL_ARTIFACT_SCHEMA, MODEL_FEATURE_SCHEMA, isJSONModelAvailable, runJSONInference, extractFeatureVector, heuristicScore, } from './score/index.js';
export { fittsMovementTime, dist2D, sampleEndpoint, preMoveSettle, clickHoldTime } from './pointer/fitts.js';
export { HumanPointer, dispatchTrajectory, dispatchKeystrokePlan, dispatchScrollPlan, systemDelay, } from './dispatch.js';
export { loadONNXModel, isONNXAvailable } from './score/onnx.js';
export { handFor, handTransition } from './keystroke/hand.js';
export { classifyDigraph, FLIGHT_TIME_PARAMS, HOLD_TIME_MEAN_MS, HOLD_TIME_STD_MS } from './keystroke/digraph.js';
// ─── Model path ──────────────────────────────────────────────────────────────
/** File URL for the bundled canonical JSON weight artifact. */
export const MODEL_JSON_URL = new URL('../model/model.json', import.meta.url);
/** File URL for the bundled ONNX artifact. */
export const MODEL_ONNX_URL = new URL('../model/model.onnx', import.meta.url);
//# sourceMappingURL=index.js.map