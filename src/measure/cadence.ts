/**
 * Cadence feature extraction.
 *
 * Measures inter-action interval CV, dwell CV, action rate.
 */

import type { CadenceFeatures, CapturedInteractionFeatures, RawInteraction } from '../types.js';

function cv(values: number[]): number {
  if (values.length < 2) return 0;
  const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
  if (mean === 0) return 0;
  const variance = values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length;
  const std = Math.sqrt(variance);
  return std / mean;
}

/**
 * Extract cadence features from a session's action timestamps.
 *
 * @param actionTimestamps  Array of action timestamps (ms, monotonic).
 * @param dwellTimes        Array of dwell times (think-time between actions, ms).
 * @returns CadenceFeatures.
 */
export function extractCadenceFeatures(
  actionTimestamps: number[],
  dwellTimes: number[],
  actionLatencies: number[] = [],
): CadenceFeatures {
  const timestamps = actionTimestamps.filter(Number.isFinite).slice().sort((a, b) => a - b);
  if (timestamps.length < 2) {
    return {
      inter_action_interval_cv: 0,
      dwell_time_cv: cv(dwellTimes.filter(Number.isFinite)),
      action_rate: 0,
      min_action_latency_ms: minimumLatency(actionLatencies),
    };
  }
  const intervals: number[] = [];
  for (let index = 1; index < timestamps.length; index++) {
    intervals.push(timestamps[index]! - timestamps[index - 1]!);
  }
  const totalDuration = timestamps[timestamps.length - 1]! - timestamps[0]!;
  return {
    inter_action_interval_cv: cv(intervals),
    dwell_time_cv: cv(dwellTimes.filter(Number.isFinite)),
    action_rate: totalDuration > 0 ? (timestamps.length / totalDuration) * 1000 : 0,
    min_action_latency_ms: minimumLatency(actionLatencies),
  };
}

function minimumLatency(values: number[]): number {
  const finiteValues = values.filter((value) => Number.isFinite(value) && value >= 0);
  return finiteValues.length > 0 ? Math.min(...finiteValues) : 0;
}

/** Reduce one session's stream; the last trusted action is consequential by default. */
export function extractCapturedInteractionFeatures(
  interactions: RawInteraction[],
  consequentialActionAtMs?: number,
): CapturedInteractionFeatures {
  const ordered = interactions
    .filter((event) => Number.isFinite(event.sinceNavigationMs))
    .slice()
    .sort((a, b) => a.sinceNavigationMs - b.sinceNavigationMs || a.capturedAt - b.capturedAt);
  const actionEvents = ordered.filter((event) => event.isTrusted && (
    event.eventType === 'click' || event.eventType === 'keydown' || event.eventType === 'wheel'
  ));
  const boundary = Number.isFinite(consequentialActionAtMs)
    ? consequentialActionAtMs!
    : actionEvents.at(-1)?.sinceNavigationMs ?? Number.POSITIVE_INFINITY;
  const priorActionCount = actionEvents.filter((event) => event.sinceNavigationMs < boundary).length;
  const actionTimestamps = actionEvents.map((event) => event.sinceNavigationMs);
  const intervals = actionTimestamps.slice(1).map((timestamp, index) => timestamp - actionTimestamps[index]!);
  const latencies = actionEvents.flatMap((event) => Number.isFinite(event.actionLatencyMs) ? [event.actionLatencyMs!] : []);
  const pointerTypes = [...new Set(ordered.map((event) => event.pointerType).filter((type) => type !== 'none'))];
  return {
    event: {
      isTrusted: ordered.length > 0 && ordered.every((event) => event.isTrusted),
      has_user_activation: ordered.some((event) => event.isTrusted && event.userActivation),
      pointerTypes,
      sequence: ordered.map((event) => event.eventType),
    },
    interaction: {
      count: priorActionCount,
      ...extractCadenceFeatures(actionTimestamps, intervals, latencies),
    },
  };
}
