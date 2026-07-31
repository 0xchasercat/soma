/**
 * Velocity profile classification and Sigma-Lognormal fit.
 *
 * Human neuromotor model: initial acceleration → peak → final deceleration.
 * Speed envelope decomposes cleanly under Sigma-Lognormal model.
 *
 * Synthetic constant-velocity or linear easing → non-bell profile, poor fit.
 *
 * Grounded in:
 *   - grimoire behav.nonhuman_velocity_profile
 *   - becaptcha-mouse-synthetic-trajectories-bot-detection.md (Sigma-Lognormal features)
 *   - intermittent-control-model-mouse-movements.md
 */
import type { TrajectoryPoint } from '../types.js';
/**
 * Classify the velocity profile shape.
 *
 * @param speeds Array of per-frame speeds (px/s).
 * @returns 'bell' | 'linear' | 'flat' | 'unknown'.
 */
export declare function classifyVelocityProfile(speeds: number[]): 'bell' | 'linear' | 'flat' | 'unknown';
/**
 * Approximate a Sigma-Lognormal decomposition with up to three non-negative
 * lognormal lobes and return bounded reconstruction R².
 *
 * The input is downsampled to 64 points so capture-scale model training remains
 * deterministic and tractable without an external numerical library.
 */
export declare function fitSigmaLognormal(speeds: number[]): number;
/**
 * Extract speed profile from trajectory.
 *
 * @param path  Trajectory (uniform temporal spacing recommended).
 * @returns Array of per-frame speeds (px/s).
 */
export declare function extractSpeeds(path: TrajectoryPoint[]): number[];
//# sourceMappingURL=velocity.d.ts.map