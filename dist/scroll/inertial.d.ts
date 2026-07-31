/**
 * Inertial scroll momentum model.
 *
 * Real inertial flicks: peak deltaY ~200–260 px (desktop trackpad) or ~200 px (touch).
 * Mochi's humanScroll peaks at only ~52 px — the validated defect from grimoire
 * scroll captures.
 *
 * Friction decay: τ ≈ 350 ms (exponential momentum loss).
 *
 * Grounded in:
 *   - grimoire behav.scroll_peak_momentum_too_weak
 *   - grimoire behav.scroll_synthesis_incoherence.yaml
 *   - captured scroll data: macos/apple/chrome-149 + ios/apple-gpu/safari-26
 */
import type { RNG } from '../rng.js';
/**
 * Generate a scroll curve. Short gestures are exact, smooth deltas; longer
 * gestures use one or more bounded exponential inertial lobes.
 */
export declare function generateInertialCurve(targetDistance: number, rng: RNG): number[];
//# sourceMappingURL=inertial.d.ts.map