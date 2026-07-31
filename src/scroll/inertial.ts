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

/** Friction decay time constant (ms). */
const DECAY_TAU_MS = 350;

/** Target peak momentum range for a real inertial flick (px). */
const PEAK_DELTA_MIN = 180;
const PEAK_DELTA_MAX = 280;

/**
 * Generate a scroll curve. Short gestures are exact, smooth deltas; longer
 * gestures use one or more bounded exponential inertial lobes.
 */
export function generateInertialCurve(targetDistance: number, rng: RNG): number[] {
  if (!Number.isFinite(targetDistance) || targetDistance === 0) return [];
  const distance = Math.abs(targetDistance);
  const sign = targetDistance < 0 ? -1 : 1;
  if (distance < PEAK_DELTA_MIN) {
    const frameCount = Math.max(1, Math.ceil(distance / 40));
    const magnitudes: number[] = [];
    let accumulated = 0;
    for (let i = 0; i < frameCount; i++) {
      const previousPhase = i / frameCount;
      const phase = (i + 1) / frameCount;
      const magnitude = distance * (phase * phase - previousPhase * previousPhase);
      magnitudes.push(magnitude);
      accumulated += magnitude;
    }
    magnitudes[magnitudes.length - 1] = distance - (accumulated - magnitudes[magnitudes.length - 1]!);
    return magnitudes.map((magnitude) => sign * magnitude);
  }

  const peakDelta = rng.nextRange(PEAK_DELTA_MIN, PEAK_DELTA_MAX);
  const dt = 1000 / 60;
  const frictionPerFrame = Math.exp(-dt / DECAY_TAU_MS);
  const magnitudes: number[] = [];
  let remaining = distance;

  // A single decaying flick can only cover a finite distance. Restart the
  // lobe after its tail falls below one pixel instead of appending an
  // unbounded residual frame.
  while (remaining > 0) {
    let velocity = peakDelta;
    while (velocity > 1 && remaining > 0) {
      const delta = Math.min(velocity, remaining);
      magnitudes.push(delta);
      remaining -= delta;
      velocity *= frictionPerFrame;
    }
  }

  return magnitudes.map((magnitude) => sign * magnitude);
}
