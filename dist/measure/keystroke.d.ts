/**
 * Keystroke feature extraction.
 *
 * Extracts CV, digraph variance, empirical CDF distance from keystroke sequences.
 */
import type { KeystrokeFeatures, RawKeystroke } from '../types.js';
/**
 * Extract keystroke features from a captured or synthesized sequence.
 *
 * @param keys        Array of key characters.
 * @param holdTimes   Array of HT per key (ms).
 * @param flightTimes Array of FT per key (ms). Length = keys.length - 1.
 * @param correctionCount Number of Backspace/Delete events.
 * @returns KeystrokeFeatures.
 */
export declare function extractKeystrokeFeatures(keys: string[], holdTimes: number[], flightTimes: number[], correctionCount: number): KeystrokeFeatures;
/** Extract keystroke evidence directly from a privacy-safe v2 capture record. */
export declare function extractCapturedKeystrokeFeatures(capture: RawKeystroke): KeystrokeFeatures;
//# sourceMappingURL=keystroke.d.ts.map