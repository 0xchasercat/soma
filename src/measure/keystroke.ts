/**
 * Keystroke feature extraction.
 *
 * Extracts CV, digraph variance, empirical CDF distance from keystroke sequences.
 */

import type { KeystrokeFeatures, RawDigraphClass, RawKeystroke } from '../types.js';
import { classifyDigraph, FLIGHT_TIME_PARAMS } from '../keystroke/digraph.js';

/**
 * Compute coefficient of variation (σ/μ).
 */
function cv(values: number[]): number {
  if (values.length < 2) return 0;
  const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
  if (mean === 0) return 0;
  const variance = values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length;
  const std = Math.sqrt(variance);
  return std / mean;
}

/**
 * Compute variance of mean FT across digraph classes.
 *
 * Human typing: FT depends on digraph → high between-class variance.
 * Machine sampling from one global distribution → near-zero variance.
 */
function varianceOfClassMeans(groups: Record<string, number[]>): number {
  const means = Object.values(groups)
    .filter((values) => values.length > 0)
    .map((values) => values.reduce((sum, value) => sum + value, 0) / values.length);
  if (means.length < 2) return 0;
  const grandMean = means.reduce((sum, mean) => sum + mean, 0) / means.length;
  return means.reduce((sum, mean) => sum + (mean - grandMean) ** 2, 0) / means.length;
}

function computeDigraphVariance(keys: string[], flightTimes: number[]): number {
  if (keys.length < 2 || flightTimes.length < 1) return 0;
  const groups: Record<string, number[]> = {
    'same-hand': [],
    'cross-hand': [],
    'after-space': [],
    'after-punct': [],
  };
  for (let index = 1; index < keys.length; index++) {
    groups[classifyDigraph(keys[index - 1]!, keys[index]!)]?.push(flightTimes[index - 1]!);
  }
  return varianceOfClassMeans(groups);
}

function computeCapturedDigraphVariance(classes: RawDigraphClass[], flightTimes: number[]): number {
  const groups: Record<string, number[]> = {
    'same-hand': [],
    'cross-hand': [],
    'after-space': [],
    'after-punct': [],
  };
  for (let index = 0; index < classes.length; index++) {
    const rawLabel = classes[index] ?? 'none';
    const label = rawLabel === 'after-punctuation' ? 'after-punct' : rawLabel;
    groups[label]?.push(flightTimes[index]!);
  }
  return varianceOfClassMeans(groups);
}

/**
 * Compute Kolmogorov-Smirnov distance between empirical CDF and a speed-scaled reference CDF.
 *
 * The reference CDF is a lognormal whose median matches the expected median flight time
 * for the observed data. This makes the metric speed-invariant: a careful typist (wpm=40)
 * and a fast typist (wpm=100) both produce human-shaped timing distributions, just with
 * different scales.
 *
 * We infer the scale from the empirical median FT, then compare shape (variance, tail weight).
 */
function computeCDFDistance(flightTimes: number[]): number {
  if (flightTimes.length < 5) return 1;

  const sorted = [...flightTimes].sort((a, b) => a - b);
  const n = sorted.length;
  const empiricalMedian = sorted[Math.floor(n / 2)]!;

  // Infer the lognormal mu from the empirical median: median = exp(mu), so mu = ln(median).
  // Use a reference sigma of 0.5 (moderate spread), typical for human digraph timing.
  const refMu = Math.log(Math.max(1, empiricalMedian));
  const refSigma = 0.5;

  // KS statistic: max |F_empirical(x) - F_reference(x)|.
  let maxDist = 0;
  for (let i = 0; i < n; i++) {
    const x = sorted[i]!;
    const empiricalCDF = (i + 1) / n;
    const z = (Math.log(Math.max(1, x)) - refMu) / refSigma;
    const referenceCDF = 0.5 * (1 + erf(z / Math.sqrt(2)));
    const dist = Math.abs(empiricalCDF - referenceCDF);
    if (dist > maxDist) maxDist = dist;
  }

  return maxDist;
}

/** Error function approximation (for normal CDF). */
function erf(x: number): number {
  // Abramowitz & Stegun approximation.
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;

  const sign = x >= 0 ? 1 : -1;
  x = Math.abs(x);

  const t = 1 / (1 + p * x);
  const y = 1 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sign * y;
}

/**
 * Extract keystroke features from a captured or synthesized sequence.
 *
 * @param keys        Array of key characters.
 * @param holdTimes   Array of HT per key (ms).
 * @param flightTimes Array of FT per key (ms). Length = keys.length - 1.
 * @param correctionCount Number of Backspace/Delete events.
 * @returns KeystrokeFeatures.
 */
export function extractKeystrokeFeatures(
  keys: string[],
  holdTimes: number[],
  flightTimes: number[],
  correctionCount: number,
): KeystrokeFeatures {
  const hold_time_cv = cv(holdTimes);
  const flight_time_cv = cv(flightTimes);
  const digraph_class_variance = computeDigraphVariance(keys, flightTimes);
  const timing_cdf_distance = computeCDFDistance(flightTimes);
  const min_flight_time = flightTimes.length > 0 ? Math.min(...flightTimes) : 0;

  return {
    hold_times: holdTimes,
    flight_times: flightTimes,
    hold_time_cv,
    flight_time_cv,
    digraph_class_variance,
    timing_cdf_distance,
    correction_count: correctionCount,
    min_flight_time,
  };
}

/** Extract keystroke evidence directly from a privacy-safe v2 capture record. */
export function extractCapturedKeystrokeFeatures(capture: RawKeystroke): KeystrokeFeatures {
  const events = capture.events
    .filter((event) => (
      event.keyKind === 'character' || event.keyKind === 'space' || event.keyKind === 'punctuation'
    ) && Number.isFinite(event.downMs) && Number.isFinite(event.upMs) && event.upMs >= event.downMs)
    .slice()
    .sort((a, b) => a.downMs - b.downMs);
  const holdTimes = events.map((event) => event.upMs - event.downMs);
  const flightTimes: number[] = [];
  const classes: RawDigraphClass[] = [];
  for (let index = 1; index < events.length; index++) {
    const flightTime = events[index]!.downMs - events[index - 1]!.downMs;
    if (!Number.isFinite(flightTime) || flightTime < 0) continue;
    flightTimes.push(flightTime);
    classes.push(events[index]!.digraphClass ?? 'none');
  }
  return {
    hold_times: holdTimes,
    flight_times: flightTimes,
    hold_time_cv: cv(holdTimes),
    flight_time_cv: cv(flightTimes),
    digraph_class_variance: computeCapturedDigraphVariance(classes, flightTimes),
    timing_cdf_distance: computeCDFDistance(flightTimes),
    correction_count: Math.max(0, capture.correctionCount),
    min_flight_time: flightTimes.length > 0 ? Math.min(...flightTimes) : 0,
  };
}
