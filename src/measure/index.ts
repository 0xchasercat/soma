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
export function measureTrajectory(path: TrajectoryPoint[]): Omit<TrajectoryFeatures, 'neural_bot_score'> {
  // Resample to uniform 60 Hz for consistent measurement.
  const resampled = resample60Hz(path);

  // Curvature.
  const curvature = computeCurvature(resampled);

  // Submovement count.
  const submovement_count = countSubmovements(resampled);

  // Velocity profile.
  const speeds = extractSpeeds(resampled);
  const velocity_profile = classifyVelocityProfile(speeds);
  const sigma_lognormal_fit = fitSigmaLognormal(speeds);

  // Micro-tremor and autocorrelation.
  const { has_micro_tremor, jitter_autocorr } = measureTremor(resampled);

  return {
    curvature,
    submovement_count,
    velocity_profile,
    sigma_lognormal_fit,
    has_micro_tremor,
    jitter_autocorr,
  };
}

export { resample60Hz, computeCurvature, countSubmovements, extractSpeeds, classifyVelocityProfile, fitSigmaLognormal, measureTremor };
export { extractKeystrokeFeatures, extractCapturedKeystrokeFeatures } from './keystroke.js';
export { extractCadenceFeatures, extractCapturedInteractionFeatures } from './cadence.js';
