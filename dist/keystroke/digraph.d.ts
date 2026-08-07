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
/** Digraph classes based on transition structure. */
export type DigraphClass = 'same-hand' | 'cross-hand' | 'after-space' | 'after-punct';
/**
 * Classify a key pair into its digraph class.
 *
 * @param prev Previous key.
 * @param curr Current key.
 * @returns Digraph class.
 */
export declare function classifyDigraph(prev: string, curr: string): DigraphClass;
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
export declare const FLIGHT_TIME_PARAMS: Record<DigraphClass, {
    mu: number;
    sigma: number;
}>;
/**
 * Hold time (key press duration) is INDEPENDENT of digraph class.
 * Human HT ~ Gaussian(80, 25) ms.
 *
 * Not lognormal — empirical data shows near-Gaussian HT distribution.
 */
export declare const HOLD_TIME_MEAN_MS = 80;
export declare const HOLD_TIME_STD_MS = 25;
/**
 * Release-to-next-press (UD) timing calibrated from trusted privacy-safe capture.
 *
 * Negative UD means rollover: the next key is pressed before the previous key is
 * released. Treating all inter-key timing as positive keydown-to-keydown latency
 * severely underproduces a common human behavior, especially across hands.
 */
export declare const RELEASE_PRESS_PARAMS: Record<DigraphClass, {
    overlapProbability: number;
    overlapMagnitude: {
        mu: number;
        sigma: number;
    };
    nonOverlap: {
        mu: number;
        sigma: number;
    };
}>;
//# sourceMappingURL=digraph.d.ts.map