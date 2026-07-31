/**
 * Keystroke timing synthesis (hold time + flight time).
 *
 * Produces per-key HT/FT sequences with realistic digraph-dependent structure.
 */
import { classifyDigraph, FLIGHT_TIME_PARAMS, HOLD_TIME_MEAN_MS, HOLD_TIME_STD_MS } from './digraph.js';
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
            const params = FLIGHT_TIME_PARAMS[dgClass];
            flightMs = Math.max(10, rng.nextLognormal(params.mu, params.sigma) * speedMult);
        }
        if (key === 'Backspace' && i > 0) {
            flightMs = rng.nextRange(200, 500);
            pauseAfterMs = rng.nextRange(100, 300);
        }
        else if (i > 0 && keys[i - 1] === 'Backspace') {
            flightMs = Math.max(10, flightMs);
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