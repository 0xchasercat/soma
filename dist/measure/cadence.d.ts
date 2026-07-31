/**
 * Cadence feature extraction.
 *
 * Measures inter-action interval CV, dwell CV, action rate.
 */
import type { CadenceFeatures, CapturedInteractionFeatures, RawInteraction } from '../types.js';
/**
 * Extract cadence features from a session's action timestamps.
 *
 * @param actionTimestamps  Array of action timestamps (ms, monotonic).
 * @param dwellTimes        Array of dwell times (think-time between actions, ms).
 * @returns CadenceFeatures.
 */
export declare function extractCadenceFeatures(actionTimestamps: number[], dwellTimes: number[], actionLatencies?: number[]): CadenceFeatures;
/** Reduce one session's stream; the last trusted action is consequential by default. */
export declare function extractCapturedInteractionFeatures(interactions: RawInteraction[], consequentialActionAtMs?: number): CapturedInteractionFeatures;
//# sourceMappingURL=cadence.d.ts.map