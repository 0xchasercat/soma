/**
 * Complete keystroke sequence synthesis.
 *
 * Composes: text → mistake injection → timing → KeystrokePlan.
 */
import type { BehaviorProfile, KeystrokePlan } from '../types.js';
/**
 * Synthesize a complete keystroke sequence for the given text.
 *
 * @param text    String to type.
 * @param profile Behavior profile (optional).
 * @param seed    RNG seed (optional — random if omitted).
 * @returns Complete keystroke plan ready to dispatch.
 */
export declare function synthesizeKeystrokes(text: string, profile?: Partial<BehaviorProfile>, seed?: number): KeystrokePlan;
//# sourceMappingURL=plan.d.ts.map