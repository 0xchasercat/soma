/**
 * Digraph class classification and per-class lognormal flight-time parameters.
 *
 * Human keystroke timing depends on the digraph structure:
 *   - same-hand transitions are slower (fingers on one hand)
 *   - cross-hand transitions are faster (hands work in parallel)
 *   - after-space and after-punctuation have distinct timing
 *
 * Detectors use digraph-class variance and empirical CDF shape to discriminate
 * human from machine. Sampling all FTs from one global distribution is detectable
 * even if the marginal ranges are correct.
 *
 * Grounded in:
 *   - grimoire behav.no_digraph_context_structure
 *   - gonzalez-2022-keystroke-liveness-detection.md
 *   - mochi-behavioral-synth.md (per-digraph-class lognormal table)
 */
import { handTransition } from './hand.js';
/**
 * Classify a key pair into its digraph class.
 *
 * @param prev Previous key.
 * @param curr Current key.
 * @returns Digraph class.
 */
export function classifyDigraph(prev, curr) {
    // Space transitions are a distinct class.
    if (prev === ' ')
        return 'after-space';
    // Punctuation transitions (non-alphanumeric, excluding space).
    if (/[^a-z0-9\s]/i.test(prev))
        return 'after-punct';
    // Hand transition.
    const trans = handTransition(prev, curr);
    if (trans === 'same')
        return 'same-hand';
    if (trans === 'cross')
        return 'cross-hand';
    // Fallback (e.g. tab → letter).
    return 'cross-hand';
}
/**
 * Lognormal parameters (μ, σ) for flight time (ms) per digraph class.
 *
 * These are the parameters of the UNDERLYING normal distribution
 * (not the mean/std of the lognormal itself).
 *
 * Derived from empirical human typing data and mochi's model:
 *   - same-hand: median ~165 ms (slower — one hand sequencing)
 *   - cross-hand: median ~121 ms (faster — hands work in parallel)
 *   - after-space: median ~200 ms (longer dwell after word boundary)
 *   - after-punct: median ~148 ms (intermediate)
 *
 * Recall: lognormal median = exp(μ).
 */
export const FLIGHT_TIME_PARAMS = {
    'same-hand': { mu: 5.1, sigma: 0.4 }, // exp(5.1) ≈ 164 ms
    'cross-hand': { mu: 4.8, sigma: 0.35 }, // exp(4.8) ≈ 121 ms
    'after-space': { mu: 5.3, sigma: 0.45 }, // exp(5.3) ≈ 200 ms
    'after-punct': { mu: 5.0, sigma: 0.4 }, // exp(5.0) ≈ 148 ms
};
/**
 * Hold time (key press duration) is INDEPENDENT of digraph class.
 * Human HT ~ Gaussian(80, 25) ms.
 *
 * Not lognormal — empirical data shows near-Gaussian HT distribution.
 */
export const HOLD_TIME_MEAN_MS = 80;
export const HOLD_TIME_STD_MS = 25;
/**
 * Release-to-next-press (UD) timing calibrated from trusted privacy-safe capture.
 *
 * Negative UD means rollover: the next key is pressed before the previous key is
 * released. Treating all inter-key timing as positive keydown-to-keydown latency
 * severely underproduces a common human behavior, especially across hands.
 */
export const RELEASE_PRESS_PARAMS = {
    'same-hand': {
        overlapProbability: 0.171,
        overlapMagnitude: { mu: Math.log(27.2), sigma: 0.72 },
        nonOverlap: { mu: Math.log(97.1), sigma: 0.58 },
    },
    'cross-hand': {
        overlapProbability: 0.485,
        overlapMagnitude: { mu: Math.log(26.6), sigma: 0.82 },
        nonOverlap: { mu: Math.log(34), sigma: 1.21 },
    },
    'after-space': {
        overlapProbability: 0.426,
        overlapMagnitude: { mu: Math.log(29.7), sigma: 0.8 },
        nonOverlap: { mu: Math.log(82.1), sigma: 1.5 },
    },
    'after-punct': {
        overlapProbability: 0.335,
        overlapMagnitude: { mu: Math.log(22.3), sigma: 0.9 },
        nonOverlap: { mu: Math.log(101.6), sigma: 1.56 },
    },
};
//# sourceMappingURL=digraph.js.map