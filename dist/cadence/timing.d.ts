/**
 * Inter-action cadence and dwell timing.
 *
 * Human sessions: variable dwell between actions, realistic action rate.
 * Metronomic or superhuman-speed sequencing is a bot tell.
 *
 * Grounded in:
 *   - grimoire behav.perfectly_sequenced_events
 *   - grimoire behav.superhuman_speed
 *   - mochi-behavioral-synth.md (per-call seeded divergence)
 */
import type { RNG } from '../rng.js';
import type { BehaviorProfile } from '../types.js';
/**
 * Generate a realistic dwell time (think-time between actions).
 * An optional cadence profile scales the result while preserving variability.
 */
export declare function generateDwell(rng: RNG, profile?: Pick<BehaviorProfile, 'cadence'>): number;
/**
 * Generate a short post-action settle delay.
 *
 * After completing an action (e.g. after a form submits), humans pause briefly
 * to observe the result before the next action.
 *
 * @param rng Seeded RNG.
 * @returns Settle time in milliseconds.
 */
export declare function generatePostActionSettle(rng: RNG): number;
/**
 * Generate an inter-action interval with optional profile cadence scaling.
 */
export declare function generateInterActionInterval(rng: RNG, profile?: Pick<BehaviorProfile, 'cadence'>): number;
//# sourceMappingURL=timing.d.ts.map