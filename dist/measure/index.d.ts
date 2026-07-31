/**
 * Unified trajectory measurement.
 *
 * Composes all trajectory feature extractors into one call.
 * Returns TrajectoryFeatures ready for scoring.
 */
import type { TrajectoryPoint, TrajectoryFeatures } from '../types.js';
import { resample60Hz } from './resample.js';
import { computeCurvature } from './curvature.js';
import { countSubmovements } from './submovements.js';
import { extractSpeeds, classifyVelocityProfile, fitSigmaLognormal } from './velocity.js';
import { measureTremor } from './tremor.js';
/**
 * Measure all trajectory features (evidence.pointer.* from grimoire).
 *
 * @param path  Raw or resampled trajectory.
 * @returns TrajectoryFeatures (excludes neural_bot_score — that comes from scorer).
 */
export declare function measureTrajectory(path: TrajectoryPoint[]): Omit<TrajectoryFeatures, 'neural_bot_score'>;
export { resample60Hz, computeCurvature, countSubmovements, extractSpeeds, classifyVelocityProfile, fitSigmaLognormal, measureTremor };
export { extractKeystrokeFeatures, extractCapturedKeystrokeFeatures } from './keystroke.js';
export { extractCadenceFeatures, extractCapturedInteractionFeatures } from './cadence.js';
//# sourceMappingURL=index.d.ts.map