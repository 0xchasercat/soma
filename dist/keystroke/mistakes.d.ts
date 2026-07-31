/**
 * QWERTY-adjacent key mistake injection.
 *
 * Humans make occasional typos — typically adjacent-key slips on QWERTY layout
 * followed by backspace correction. Perfectly error-free typing across long
 * awkward strings is a weak but real machine tell.
 *
 * Pattern: type wrong adjacent key → delay 200–500 ms → Backspace → delay 100–300 ms → correct key.
 *
 * Grounded in:
 *   - grimoire behav.no_typing_errors_or_corrections
 *   - mochi-behavioral-synth.md (mistakeRate=0.02, adjacent-key injection)
 */
import type { RNG } from '../rng.js';
/**
 * Pick a realistic adjacent-key mistake for the given key.
 *
 * @param key Target key.
 * @param rng Seeded RNG.
 * @returns An adjacent key, or the original key if no adjacency defined.
 */
export declare function pickAdjacentKey(key: string, rng: RNG): string;
/**
 * Inject realistic QWERTY-adjacent mistakes into a key sequence.
 *
 * Each eligible key has a mistakeRate probability of being replaced with:
 *   [wrongKey, Backspace, correctKey]
 * with realistic delays between them.
 *
 * @param keys        Original key sequence.
 * @param mistakeRate Probability per character (default: 0.02 ≈ 2%).
 * @param rng         Seeded RNG.
 * @returns Expanded key sequence with injected mistakes + corrections.
 */
export declare function injectMistakes(keys: string[], mistakeRate: number, rng: RNG): string[];
//# sourceMappingURL=mistakes.d.ts.map