/**
 * Soma — behavioral synthesis and measurement primitives grounded in the
 * behavioral contracts documented by clandestine-grimoire.
 *
 * The library provides deterministic plans and local measurement/scoring. It
 * does not claim that generated input passes any third-party detector.
 */

// ─── Core Types ──────────────────────────────────────────────────────────────

export type {
  Point2D,
  TrajectoryPoint,
  TargetBox,
  BehaviorProfile,
  TrajectoryPlan,
  KeyEvent,
  KeystrokePlan,
  ScrollFrame,
  ScrollPlan,
  TrajectoryFeatures,
  KeystrokeFeatures,
  CadenceFeatures,
  ModelFeatureVector,
  CaptureSourceSchema,
  RawKeyHand,
  RawKeyKind,
  RawDigraphClass,
  RawInteraction,
  CapturedInteractionFeatures,
  RawMovement,
  RawKeystroke,
  RawScroll,
  CaptureExport,
  LabeledSample,
} from './types.js';

// ─── RNG & Profile ───────────────────────────────────────────────────────────

export { RNG, defaultRNG } from './rng.js';
export { DEFAULT_PROFILE, PROFILES, mergeProfile, clampProfile } from './profile.js';

// ─── Synthesis ───────────────────────────────────────────────────────────────

export { synthesizeMovement, synthesizeClick } from './pointer/trajectory.js';
export { synthesizeKeystrokes } from './keystroke/plan.js';
export { synthesizeScroll } from './scroll/plan.js';
export {
  synthesizeMovementFlow,
  synthesizeClickFlow,
  loadFlowModel,
  validateFlowStats,
  reconstructGesture,
} from './pointer/flow-synthesis.js';
export type {
  FlowModel,
  FlowStats,
  FlowSession,
  FlowTensor,
  FlowTensorFactory,
} from './pointer/flow-synthesis.js';
export { generateDwell, generatePostActionSettle, generateInterActionInterval, generatePageReadDwell } from './cadence/timing.js';

// ─── Measurement ─────────────────────────────────────────────────────────────

export {
  measureTrajectory,
  resample60Hz,
  computeCurvature,
  countSubmovements,
  extractSpeeds,
  classifyVelocityProfile,
  fitSigmaLognormal,
  measureTremor,
  extractKeystrokeFeatures,
  extractCapturedKeystrokeFeatures,
  extractCadenceFeatures,
  extractCapturedInteractionFeatures,
} from './measure/index.js';

// ─── Scoring ─────────────────────────────────────────────────────────────────

export {
  scoreTrajectory,
  measureAndScore,
  loadJSONModel,
  loadJSONModelText,
  validateModelWeights,
  validateModelArtifact,
  MODEL_ARTIFACT_SCHEMA,
  MODEL_FEATURE_SCHEMA,
  isJSONModelAvailable,
  runJSONInference,
  extractFeatureVector,
  heuristicScore,
} from './score/index.js';

export { fittsMovementTime, dist2D, sampleEndpoint, preMoveSettle, clickHoldTime } from './pointer/fitts.js';
export {
  HumanPointer,
  dispatchTrajectory,
  dispatchKeystrokePlan,
  dispatchScrollPlan,
  systemDelay,
} from './dispatch.js';
export type {
  Delay,
  PointerMoveDriver,
  PointerClickDriver,
  KeyboardDriver,
  ScrollDriver,
  DispatchedClick,
} from './dispatch.js';
export { loadONNXModel, isONNXAvailable } from './score/onnx.js';
export { handFor, handTransition } from './keystroke/hand.js';
export { classifyDigraph, FLIGHT_TIME_PARAMS, HOLD_TIME_MEAN_MS, HOLD_TIME_STD_MS } from './keystroke/digraph.js';

// ─── Model path ──────────────────────────────────────────────────────────────

/** File URL for the bundled canonical JSON weight artifact. */
export const MODEL_JSON_URL: URL = new URL('../model/model.json', import.meta.url);

/** File URL for the bundled ONNX artifact. */
export const MODEL_ONNX_URL: URL = new URL('../model/model.onnx', import.meta.url);

/** File URL for the bundled conditional-flow generator (pointer synthesis). */
export const FLOW_ONNX_URL: URL = new URL('../model/flow.onnx', import.meta.url);

/** File URL for the flow's normalization stats. Required alongside FLOW_ONNX_URL. */
export const FLOW_STATS_URL: URL = new URL('../model/flow.stats.json', import.meta.url);