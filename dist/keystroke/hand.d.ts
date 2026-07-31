/**
 * QWERTY hand classifier.
 *
 * Maps each key to 'left' | 'right' hand based on standard touch-typing
 * home row position. Used for digraph-class classification.
 *
 * Grounded in:
 *   - grimoire behav.no_digraph_context_structure
 *   - mochi-behavioral-synth.md (handFor function)
 */
/**
 * Classify which hand types a given key.
 *
 * @param key Single character (lowercase/uppercase/digit/punctuation).
 * @returns 'left' | 'right' | 'neutral'.
 *          'neutral' = space, tab, enter, backspace, etc. (no hand assignment).
 */
export declare function handFor(key: string): 'left' | 'right' | 'neutral';
/**
 * Classify whether two consecutive keys use the same hand.
 *
 * @param a First key.
 * @param b Second key.
 * @returns 'same' | 'cross' | 'neutral'.
 *          'neutral' if either key is neutral (space, tab, etc.).
 */
export declare function handTransition(a: string, b: string): 'same' | 'cross' | 'neutral';
//# sourceMappingURL=hand.d.ts.map