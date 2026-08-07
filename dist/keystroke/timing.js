/**
 * Keystroke timing synthesis (hold time + flight time).
 *
 * Produces per-key HT/FT sequences with realistic digraph-dependent structure.
 */
import { classifyDigraph, HOLD_TIME_MEAN_MS, HOLD_TIME_STD_MS, RELEASE_PRESS_PARAMS } from './digraph.js';
/**
 * Synthesize realistic keystroke timing for a sequence of keys.
 *
 * @param keys    Array of key characters (e.g. ['h', 'e', 'l', 'l', 'o']).
 * @param profile Behavior profile (wpm scales the baseline flight times).
 * @param rng     Seeded RNG.
 * @returns Array of KeyTiming (one per key).
 */
export function synthesizeTiming(keys, profile, rng) {
    if (keys.length === 0)
        return [];
    // Derive speed multiplier from wpm.
    // Average word = 5 chars. wpm=65 → 325 chars/min → ~5.4 chars/sec → ~185 ms/char.
    // Baseline (wpm=65) → speedMult=1.0.
    // Higher wpm → lower speedMult (faster typing).
    const baselineWpm = 65;
    const speedMult = baselineWpm / profile.wpm;
    const timings = [];
    for (let i = 0; i < keys.length; i++) {
        const key = keys[i];
        const holdMs = Math.max(20, rng.nextGaussian(HOLD_TIME_MEAN_MS, HOLD_TIME_STD_MS));
        let flightMs = 0;
        let pauseAfterMs;
        if (i > 0) {
            const prevKey = keys[i - 1];
            const dgClass = classifyDigraph(prevKey, key);
            const params = RELEASE_PRESS_PARAMS[dgClass];
            const previousHoldMs = timings[i - 1].holdMs;
            // Model key-up -> next-key-down directly. Negative values preserve rollover;
            // DD (the public flightMs field) is then derived from the previous hold.
            const speedAdjustedOverlap = Math.min(0.85, Math.max(0.02, params.overlapProbability * Math.sqrt(profile.wpm / baselineWpm)));
            const releasePressMs = rng.next() < speedAdjustedOverlap
                ? -rng.nextLognormal(params.overlapMagnitude.mu, params.overlapMagnitude.sigma)
                : rng.nextLognormal(params.nonOverlap.mu, params.nonOverlap.sigma) * speedMult;
            flightMs = Math.max(8, previousHoldMs + releasePressMs);
        }
        if (key === 'Backspace' && i > 0) {
            flightMs = rng.nextRange(200, 500);
            pauseAfterMs = rng.nextRange(100, 300);
        }
        else if (i > 0 && keys[i - 1] === 'Backspace') {
            flightMs = Math.max(32, flightMs);
        }
        timings.push({ key, holdMs, flightMs, ...(pauseAfterMs === undefined ? {} : { pauseAfterMs }) });
    }
    return timings;
}
/** Convert relative timing into absolute event timestamps. */
export function timingsToAbsolute(timings) {
    let cumulative = 0;
    return timings.map((t) => {
        const downMs = cumulative + t.flightMs;
        const upMs = downMs + t.holdMs;
        cumulative = downMs + (t.pauseAfterMs ?? 0);
        return {
            key: t.key,
            downMs,
            upMs,
            ...(t.pauseAfterMs === undefined ? {} : { pauseAfterMs: t.pauseAfterMs }),
        };
    });
}
//# sourceMappingURL=timing.js.map