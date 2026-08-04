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
 *   - 494MB real human capture (2026-08-04): inter-click dwell p50=283ms, p95=4883ms, CV p50=1.06
 */

import type { RNG } from '../rng.js';
import type { BehaviorProfile } from '../types.js';

/**
 * Generate a realistic between-action dwell time.
 *
 * Fitted to 52,326 real inter-click gaps from the 494MB capture
 * (p50=283ms, p75=949ms, p95=4883ms, per-session CV p50=1.06) by grid search over
 * a three-component lognormal mixture. Resulting bucket shares vs real:
 *
 *   <1s      77.5%  (real 75.7%)   rapid sequential navigation
 *   1-3s     14.8%  (real 17.2%)   scanning
 *   3-10s     5.3%  (real  4.0%)   reading a snippet
 *   10-30s    1.0%  (real  1.4%)   engaged
 *   30-120s   1.1%  (real  0.9%)   long click
 *   >120s     0.2%  (real  0.8%)   deep read
 *
 * The third component matters beyond kinematics: a session with no long dwell at
 * all is uniform-engagement, which is what the Navboost-class goodClicks /
 * lastLongestClicks aggregates are built to separate from real interest.
 */
export function generateDwell(rng: RNG, profile?: Pick<BehaviorProfile, 'cadence'>): number {
  const cadence = Number.isFinite(profile?.cadence) ? Math.max(0.25, Math.min(4, profile!.cadence!)) : 1;
  const u = rng.next();
  const raw = u < 0.025
    ? rng.nextLognormal(10.3, 1.0)   // long engagement (30s+)
    : u < 0.245
      ? rng.nextLognormal(7.6, 0.55) // scan pause (~2s)
      : rng.nextLognormal(5.45, 0.55); // rapid navigation (~233ms)
  return Math.max(100, raw / cadence);
}

/**
 * Generate a dwell for reading a content page (article, search result).
 *
 * Distinct from between-action dwell: this models how long a user spends
 * on a page before leaving. The distribution is bimodal — most result
 * visits are quick (scan + leave), but a meaningful fraction are long reads.
 *
 * Shape derived from real inter-click gaps with Navboost context:
 *   - lastLongestClicks: sessions with ≥1 long dwell (>30s) score higher
 *   - badClicks: bounces (<5s) score lower; 4% of real gaps are 3-10s, 2.3% >30s
 *
 * @param pageKind - The type of content being read.
 * @param rng - Seeded RNG.
 */
export function generatePageReadDwell(
  pageKind: 'article' | 'search_result' | 'landing_page' | 'quick_fact',
  rng: RNG,
): number {
  switch (pageKind) {
    case 'article':
      // Articles: bimodal — 40% quick scan (5-15s), 60% real read (30-120s).
      return rng.next() < 0.40
        ? rng.nextLognormal(8.7, 0.4)     // p50≈6000ms: quick scan
        : rng.nextLognormal(10.5, 0.7);   // p50≈36000ms: real read (lastLongestClicks bucket)
    case 'search_result':
      // SERP result: mostly quick engagement (3-20s), occasionally deeper.
      return rng.next() < 0.70
        ? rng.nextLognormal(8.2, 0.5)     // p50≈3600ms: skim result
        : rng.nextLognormal(9.5, 0.8);    // p50≈13000ms: engaged read
    case 'quick_fact':
      // Weather/conversion/time: visit is answered immediately.
      return rng.nextLognormal(7.5, 0.4); // p50≈1800ms
    case 'landing_page':
      // General page: moderate engagement.
      return rng.nextLognormal(9.0, 0.9); // p50≈8100ms
    default:
      return generateDwell(rng);
  }
}

/**
 * Generate a short post-action settle delay.
 *
 * After completing an action (e.g. after a form submits), humans pause briefly
 * to observe the result before the next action.
 */
export function generatePostActionSettle(rng: RNG): number {
  // Gaussian(200, 60) ms — tighter than before to match real p25=208ms.
  return Math.max(80, rng.nextGaussian(200, 60));
}

/**
 * Generate an inter-action interval with optional profile cadence scaling.
 */
export function generateInterActionInterval(rng: RNG, profile?: Pick<BehaviorProfile, 'cadence'>): number {
  const cadence = Number.isFinite(profile?.cadence) ? Math.max(0.25, Math.min(4, profile!.cadence!)) : 1;
  const dwell = generateDwell(rng, profile);
  const settle = rng.nextGaussian(100, 40) / cadence;
  return Math.max(150 / cadence, dwell + settle);
}
