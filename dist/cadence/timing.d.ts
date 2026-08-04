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
export declare function generateDwell(rng: RNG, profile?: Pick<BehaviorProfile, 'cadence'>): number;
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
export declare function generatePageReadDwell(pageKind: 'article' | 'search_result' | 'landing_page' | 'quick_fact', rng: RNG): number;
/**
 * Generate a short post-action settle delay.
 *
 * After completing an action (e.g. after a form submits), humans pause briefly
 * to observe the result before the next action.
 */
export declare function generatePostActionSettle(rng: RNG): number;
/**
 * Generate an inter-action interval with optional profile cadence scaling.
 */
export declare function generateInterActionInterval(rng: RNG, profile?: Pick<BehaviorProfile, 'cadence'>): number;
//# sourceMappingURL=timing.d.ts.map