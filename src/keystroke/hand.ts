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

/** Left hand owns these keys (US-QWERTY layout, touch-typing home row). */
const LEFT_KEYS = new Set([
  // Letters.
  'q', 'w', 'e', 'r', 't',
  'a', 's', 'd', 'f', 'g',
  'z', 'x', 'c', 'v', 'b',
  // Numbers (left side).
  '1', '2', '3', '4', '5',
  // Shifted versions.
  'Q', 'W', 'E', 'R', 'T',
  'A', 'S', 'D', 'F', 'G',
  'Z', 'X', 'C', 'V', 'B',
  '!', '@', '#', '$', '%',
  // Special.
  '`', '~',
]);

/** Right hand owns these keys. */
const RIGHT_KEYS = new Set([
  // Letters.
  'y', 'u', 'i', 'o', 'p',
  'h', 'j', 'k', 'l',
  'n', 'm',
  // Numbers (right side).
  '6', '7', '8', '9', '0',
  // Shifted versions.
  'Y', 'U', 'I', 'O', 'P',
  'H', 'J', 'K', 'L',
  'N', 'M',
  '^', '&', '*', '(', ')',
  // Punctuation (right side).
  ';', ':', "'", '"',
  '[', '{', ']', '}',
  '\\', '|',
  '-', '_', '=', '+',
]);

/**
 * Classify which hand types a given key.
 *
 * @param key Single character (lowercase/uppercase/digit/punctuation).
 * @returns 'left' | 'right' | 'neutral'.
 *          'neutral' = space, tab, enter, backspace, etc. (no hand assignment).
 */
export function handFor(key: string): 'left' | 'right' | 'neutral' {
  if (LEFT_KEYS.has(key)) return 'left';
  if (RIGHT_KEYS.has(key)) return 'right';
  return 'neutral';
}

/**
 * Classify whether two consecutive keys use the same hand.
 *
 * @param a First key.
 * @param b Second key.
 * @returns 'same' | 'cross' | 'neutral'.
 *          'neutral' if either key is neutral (space, tab, etc.).
 */
export function handTransition(a: string, b: string): 'same' | 'cross' | 'neutral' {
  const ha = handFor(a);
  const hb = handFor(b);
  if (ha === 'neutral' || hb === 'neutral') return 'neutral';
  return ha === hb ? 'same' : 'cross';
}
