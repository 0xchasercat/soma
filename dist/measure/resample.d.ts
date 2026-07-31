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
export declare function resample60Hz(path: TrajectoryPoint[]): TrajectoryPoint[];
/**
 * Resample a trajectory to a fixed number of points (uniform in time).
 *
 * @param path        Input trajectory.
 * @param targetCount Desired number of samples.
 * @returns Resampled trajectory.
 */
export declare function resampleToCount(path: TrajectoryPoint[], targetCount: number): TrajectoryPoint[];
//# sourceMappingURL=resample.d.ts.map