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
/** QWERTY adjacency map (physical neighbors for common mistake simulation). */
const ADJACENT_KEYS = {
    a: ['q', 's', 'z'],
    b: ['v', 'g', 'h', 'n'],
    c: ['x', 'd', 'f', 'v'],
    d: ['s', 'e', 'r', 'f', 'c', 'x'],
    e: ['w', 'r', 'd', 's'],
    f: ['d', 'r', 't', 'g', 'v', 'c'],
    g: ['f', 't', 'y', 'h', 'b', 'v'],
    h: ['g', 'y', 'u', 'j', 'n', 'b'],
    i: ['u', 'o', 'k', 'j'],
    j: ['h', 'u', 'i', 'k', 'n', 'm'],
    k: ['j', 'i', 'o', 'l', 'm'],
    l: ['k', 'o', 'p'],
    m: ['n', 'j', 'k'],
    n: ['b', 'h', 'j', 'm'],
    o: ['i', 'p', 'l', 'k'],
    p: ['o', 'l'],
    q: ['w', 'a'],
    r: ['e', 't', 'f', 'd'],
    s: ['a', 'w', 'e', 'd', 'x', 'z'],
    t: ['r', 'y', 'g', 'f'],
    u: ['y', 'i', 'j', 'h'],
    v: ['c', 'f', 'g', 'b'],
    w: ['q', 'e', 's', 'a'],
    x: ['z', 's', 'd', 'c'],
    y: ['t', 'u', 'h', 'g'],
    z: ['a', 's', 'x'],
    // Numbers (top row).
    '1': ['2', 'q'],
    '2': ['1', '3', 'q', 'w'],
    '3': ['2', '4', 'w', 'e'],
    '4': ['3', '5', 'e', 'r'],
    '5': ['4', '6', 'r', 't'],
    '6': ['5', '7', 't', 'y'],
    '7': ['6', '8', 'y', 'u'],
    '8': ['7', '9', 'u', 'i'],
    '9': ['8', '0', 'i', 'o'],
    '0': ['9', 'o', 'p'],
};
/**
 * Pick a realistic adjacent-key mistake for the given key.
 *
 * @param key Target key.
 * @param rng Seeded RNG.
 * @returns An adjacent key, or the original key if no adjacency defined.
 */
export function pickAdjacentKey(key, rng) {
    const lowerKey = key.toLowerCase();
    const adj = ADJACENT_KEYS[lowerKey];
    if (!adj || adj.length === 0)
        return key; // no adjacency → no mistake
    // Pick a random adjacent key.
    const wrongKey = adj[rng.nextInt(adj.length)];
    // Preserve case if original was uppercase.
    return key === key.toUpperCase() && key !== key.toLowerCase()
        ? wrongKey.toUpperCase()
        : wrongKey;
}
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
export function injectMistakes(keys, mistakeRate, rng) {
    const out = [];
    for (const key of keys) {
        // Skip non-letter keys (space, punctuation, etc.) — no adjacency.
        const isLetter = /[a-z0-9]/i.test(key);
        if (!isLetter || rng.next() >= mistakeRate) {
            // No mistake.
            out.push(key);
            continue;
        }
        // Inject mistake sequence: wrongKey → Backspace → correctKey.
        const wrongKey = pickAdjacentKey(key, rng);
        if (wrongKey === key) {
            // No adjacency available.
            out.push(key);
            continue;
        }
        out.push(wrongKey, 'Backspace', key);
    }
    return out;
}
//# sourceMappingURL=mistakes.js.map