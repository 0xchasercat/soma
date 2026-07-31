/**
 * Complete keystroke sequence synthesis.
 *
 * Composes: text → mistake injection → timing → KeystrokePlan.
 */

import type { BehaviorProfile, KeystrokePlan, KeyEvent } from '../types.js';
import { RNG } from '../rng.js';
import { mergeProfile } from '../profile.js';
import { injectMistakes } from './mistakes.js';
import { synthesizeTiming, timingsToAbsolute } from './timing.js';

/** Physical keyboard descriptor. */
interface KeyDescriptor {
  code: string;
  shiftKey: boolean;
}

/** Map printable characters to US keyboard physical code and modifiers. */
function describeKey(char: string): KeyDescriptor {
  if (/^[a-z]$/i.test(char)) return { code: `Key${char.toUpperCase()}`, shiftKey: char !== char.toLowerCase() };
  if (/^[0-9]$/.test(char)) return { code: `Digit${char}`, shiftKey: false };
  const unshifted: Record<string, string> = {
    ' ': 'Space', '\t': 'Tab', '\n': 'Enter', '.': 'Period', ',': 'Comma', ';': 'Semicolon',
    "'": 'Quote', '[': 'BracketLeft', ']': 'BracketRight', '-': 'Minus', '=': 'Equal',
    '/': 'Slash', '\\': 'Backslash', '`': 'Backquote',
  };
  const shifted: Record<string, string> = {
    '~': 'Backquote', '!': 'Digit1', '@': 'Digit2', '#': 'Digit3', '$': 'Digit4', '%': 'Digit5',
    '^': 'Digit6', '&': 'Digit7', '*': 'Digit8', '(': 'Digit9', ')': 'Digit0', '_': 'Minus',
    '+': 'Equal', '{': 'BracketLeft', '}': 'BracketRight', '|': 'Backslash', ':': 'Semicolon',
    '"': 'Quote', '<': 'Comma', '>': 'Period', '?': 'Slash',
  };
  if (char === 'Backspace') return { code: 'Backspace', shiftKey: false };
  if (unshifted[char]) return { code: unshifted[char]!, shiftKey: false };
  if (shifted[char]) return { code: shifted[char]!, shiftKey: true };
  return { code: 'Unidentified', shiftKey: false };
}

/**
 * Synthesize a complete keystroke sequence for the given text.
 *
 * @param text    String to type.
 * @param profile Behavior profile (optional).
 * @param seed    RNG seed (optional — random if omitted).
 * @returns Complete keystroke plan ready to dispatch.
 */
export function synthesizeKeystrokes(
  text: string,
  profile?: Partial<BehaviorProfile>,
  seed?: number,
): KeystrokePlan {
  const p = mergeProfile(profile);
  const rng = new RNG(seed);

  // Split text into individual characters.
  const originalKeys = text.split('');

  // Inject realistic QWERTY-adjacent mistakes at mistakeRate.
  const keysWithMistakes = injectMistakes(originalKeys, p.mistakeRate, rng);

  // Synthesize per-key timing (HT + digraph-dependent FT).
  const timings = synthesizeTiming(keysWithMistakes, p, rng);

  // Convert to absolute timestamps.
  const absolute = timingsToAbsolute(timings);

  const events: KeyEvent[] = absolute.map((a, index) => {
    const descriptor = describeKey(a.key);
    const correction = a.key === 'Backspace'
      ? { kind: 'backspace' as const, pauseMs: a.pauseAfterMs ?? 0 }
      : index > 0 && absolute[index - 1]!.key === 'Backspace'
        ? { kind: 'retry' as const, pauseMs: a.pauseAfterMs ?? 0 }
        : undefined;
    return {
      key: a.key,
      code: descriptor.code,
      downMs: a.downMs,
      holdMs: a.upMs - a.downMs,
      shiftKey: descriptor.shiftKey,
      modifiers: { shift: descriptor.shiftKey },
      ...(a.pauseAfterMs === undefined ? {} : { pauseAfterMs: a.pauseAfterMs }),
      ...(correction === undefined ? {} : { correction }),
    };
  });

  const totalDurationMs = events.length > 0
    ? events[events.length - 1]!.downMs + events[events.length - 1]!.holdMs
    : 0;

  return {
    events,
    totalDurationMs,
  };
}
