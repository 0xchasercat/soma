/**
 * Resample trajectory to uniform 60 Hz temporal spacing.
 *
 * Measurement and scoring require consistent sample rates. The browser's native
 * mousemove rate varies; resampling normalizes it for feature extraction.
 */

import type { TrajectoryPoint } from '../types.js';

/**
 * Resample a trajectory to exactly 60 Hz (16.67 ms per frame).
 *
 * @param path  Input trajectory (variable rate).
 * @returns Resampled trajectory at 60 Hz.
 */
export function resample60Hz(path: TrajectoryPoint[]): TrajectoryPoint[] {
  if (path.length === 0) return [];
  const finitePath = path
    .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.tMs))
    .slice()
    .sort((a, b) => a.tMs - b.tMs);
  if (finitePath.length < 2) return finitePath.map((p) => ({ ...p }));
  const first = finitePath[0]!;
  const last = finitePath[finitePath.length - 1]!;
  const duration = last.tMs - first.tMs;
  if (!(duration > 0)) return [{ ...first }];

  const step = 1000 / 60;
  const targetTimes: number[] = [];
  for (let t = first.tMs; t < last.tMs; t += step) targetTimes.push(t);
  targetTimes.push(last.tMs);
  const out: TrajectoryPoint[] = [];
  let j = 1;
  for (const targetT of targetTimes) {
    while (j < finitePath.length - 1 && finitePath[j]!.tMs < targetT) j++;
    const p0 = finitePath[j - 1]!;
    const p1 = finitePath[j]!;
    const dt = p1.tMs - p0.tMs;
    const u = dt > 0 ? Math.max(0, Math.min(1, (targetT - p0.tMs) / dt)) : 0;
    out.push({ x: p0.x + (p1.x - p0.x) * u, y: p0.y + (p1.y - p0.y) * u, tMs: targetT });
  }
  out[out.length - 1] = { x: last.x, y: last.y, tMs: last.tMs };
  return out;
}

/**
 * Resample a trajectory to a fixed number of points (uniform in time).
 *
 * @param path        Input trajectory.
 * @param targetCount Desired number of samples.
 * @returns Resampled trajectory.
 */
export function resampleToCount(path: TrajectoryPoint[], targetCount: number): TrajectoryPoint[] {
  if (path.length < 2 || targetCount <= 1) return path.slice(0, Math.min(targetCount, path.length));

  const first = path[0]!;
  const last = path[path.length - 1]!;
  const duration = last.tMs - first.tMs;

  if (duration <= 0) return [first];

  const frameDt = duration / (targetCount - 1);
  const out: TrajectoryPoint[] = [];

  for (let i = 0; i < targetCount; i++) {
    const targetT = first.tMs + i * frameDt;

    let j = path.findIndex((p) => p.tMs >= targetT);
    if (j <= 0) j = 1;

    const p0 = path[j - 1]!;
    const p1 = path[j]!;
    const dt = p1.tMs - p0.tMs;
    const t = dt > 0 ? (targetT - p0.tMs) / dt : 0;

    out.push({
      x: p0.x + (p1.x - p0.x) * t,
      y: p0.y + (p1.y - p0.y) * t,
      tMs: targetT,
    });
  }

  return out;
}
