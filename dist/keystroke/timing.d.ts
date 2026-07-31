/**
 * Keystroke timing synthesis (hold time + flight time).
 *
 * Produces per-key HT/FT sequences with realistic digraph-dependent structure.
 */
import type { RNG } from '../rng.js';
import type { BehaviorProfile } from '../types.js';
/** Timing for one keystroke. */
export interface KeyTiming {
    key: string;
    holdMs: number;
    flightMs: number;
    /** Explicit pause after this event before the next keydown. */
    pauseAfterMs?: number;
}
/**
 * Synthesize realistic keystroke timing for a sequence of keys.
 *
 * @param keys    Array of key characters (e.g. ['h', 'e', 'l', 'l', 'o']).
 * @param profile Behavior profile (wpm scales the baseline flight times).
 * @param rng     Seeded RNG.
 * @returns Array of KeyTiming (one per key).
 */
export declare function synthesizeTiming(keys: string[], profile: BehaviorProfile, rng: RNG): KeyTiming[];
/** Convert relative timing into absolute event timestamps. */
export declare function timingsToAbsolute(timings: KeyTiming[]): Array<{
    key: string;
    downMs: number;
    upMs: number;
    pauseAfterMs?: number;
}>;
//# sourceMappingURL=timing.d.ts.map