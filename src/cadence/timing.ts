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
export function generateDwell(rng: RNG, profile?: Pick<BehaviorProfile, 'cadence'>): number {
  const cadence = Number.isFinite(profile?.cadence) ? Math.max(0.25, Math.min(4, profile!.cadence!)) : 1;
  return Math.max(200, rng.nextLognormal(6.5, 0.6) / cadence);
}

/**
 * Generate a short post-action settle delay.
 *
 * After completing an action (e.g. after a form submits), humans pause briefly
 * to observe the result before the next action.
 *
 * @param rng Seeded RNG.
 * @returns Settle time in milliseconds.
 */
export function generatePostActionSettle(rng: RNG): number {
  // Gaussian(300, 80) ms.
  return Math.max(100, rng.nextGaussian(300, 80));
}

/**
 * Generate an inter-action interval with optional profile cadence scaling.
 */
export function generateInterActionInterval(rng: RNG, profile?: Pick<BehaviorProfile, 'cadence'>): number {
  const cadence = Number.isFinite(profile?.cadence) ? Math.max(0.25, Math.min(4, profile!.cadence!)) : 1;
  const dwell = generateDwell(rng, profile);
  const settle = rng.nextGaussian(150, 50) / cadence;
  return Math.max(250 / cadence, dwell + settle);
}
