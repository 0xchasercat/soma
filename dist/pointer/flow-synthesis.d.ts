/**
 * Flow-based pointer synthesis.
 *
 * Replaces the parametric Bezier + Fitts + overshoot pipeline with a sample
 * from a conditional normalizing flow trained on 19,755 real captured gestures
 * (see flow/train_flow.py). The flow models
 *
 *     p( du[0..126], dv[0..126], log(duration_ms) | log(distance), log(target_size) )
 *
 * in a canonical target-relative frame: start at the origin, endpoint on the
 * +X axis at unit distance, 129 points on a uniform time grid.
 *
 * Why this beats the parametric path
 * ----------------------------------
 * Overshoot rate, peak-velocity timing and velocity-profile shape are learned
 * jointly from data instead of being hand-tuned one constant at a time. The
 * Phase 3 audit measured discriminator separability dropping from 91.1% to
 * 31.6% (overshoot), 87.3% to 29.5% (peak timing) and 84.7% to 0.0%
 * (bell profile).
 *
 * Post-processing is NOT cosmetic
 * -------------------------------
 * The same audit found three defects the raw samples do not fix on their own,
 * each of which is independently a detection signal:
 *
 *   1. terminal velocity   — raw samples land at 72% of peak speed; real
 *                            gestures land at 0.4%. A cursor that arrives at
 *                            full speed and stops dead is not physical.
 *   2. submovement count    — raw samples show ~40 speed reversals against a
 *                            real ~13, because per-timestep spline noise reads
 *                            as high-frequency chatter.
 *   3. lateral tremor band  — raw high-frequency lateral energy is ~9x real.
 *
 * `applyDecelerationTaper` and `smoothLateral` below correct 1-3 in the
 * reconstruction, which is the same code path the Phase 3 audit measures, so
 * the numbers reported there describe what actually ships.
 *
 * Determinism
 * -----------
 * z is drawn from the caller's seeded RNG, ONNX inference is deterministic, and
 * everything after it is pure arithmetic. Same seed plus same target implies a
 * byte-identical TrajectoryPlan, which is what sip-eval's recovery model needs.
 */
import type { BehaviorProfile, Point2D, TargetBox, TrajectoryPlan } from '../types.js';
/** Normalization constants and layout emitted by flow/export_stats.py. */
export interface FlowStats {
    schemaVersion: number;
    arch: {
        dataDim: number;
        contextDim: number;
        nTransforms: number;
        nBins: number;
        tailBound: number;
    };
    layout: {
        nSteps: number;
        nFree: number;
        duStart: number;
        dvStart: number;
        logDurationIndex: number;
    };
    stats: {
        xMean: number[];
        xStd: number[];
        cMean: number[];
        cStd: number[];
    };
    bestValNll: number;
}
/**
 * Minimal ONNX runtime surface used here.
 *
 * Declared structurally so `onnxruntime-node` stays an optional peer
 * dependency: consumers that never call the flow path do not need it
 * installed, and `soma` still typechecks without it.
 */
export interface FlowSession {
    run(feeds: Record<string, FlowTensor>): Promise<Record<string, {
        data: Float32Array;
    }>>;
}
export interface FlowTensor {
    readonly data: Float32Array;
    readonly dims: readonly number[];
}
/** Constructs a runtime tensor; pass `(d, dims) => new ort.Tensor('float32', d, dims)`. */
export type FlowTensorFactory = (data: Float32Array, dims: number[]) => FlowTensor;
/** A loaded flow model: session, normalization stats, and tensor constructor. */
export interface FlowModel {
    session: FlowSession;
    stats: FlowStats;
    tensor: FlowTensorFactory;
}
/** Throw unless the stats artifact is internally consistent. */
export declare function validateFlowStats(stats: FlowStats): void;
/**
 * Load a flow model.
 *
 * `createSession` and `tensor` are injected so this module never imports
 * `onnxruntime-node` directly — consumers that do not use the flow path do
 * not pay the dependency.
 *
 * @example
 * ```ts
 * import * as ort from 'onnxruntime-node';
 * const model = await loadFlowModel({
 *   onnxPath: fileURLToPath(FLOW_ONNX_URL),
 *   statsJson: await Bun.file(fileURLToPath(FLOW_STATS_URL)).text(),
 *   createSession: (p) => ort.InferenceSession.create(p) as never,
 *   tensor: (d, dims) => new ort.Tensor('float32', d, dims) as never,
 * });
 * ```
 */
export declare function loadFlowModel(options: {
    onnxPath: string;
    statsJson: string;
    createSession: (path: string) => Promise<FlowSession>;
    tensor: FlowTensorFactory;
}): Promise<FlowModel>;
/**
 * Turn one standardized model vector into a canonical-frame gesture.
 *
 * Mirrors `reconstruct()` in flow/audit_eval.py. Exported so the audit and the
 * runtime can be checked against each other.
 */
export declare function reconstructGesture(xStandardized: Float32Array, stats: FlowStats): {
    u: Float64Array;
    v: Float64Array;
    t: Float64Array;
    durationMs: number;
};
/**
 * Synthesize a pointer movement by sampling the flow.
 *
 * Drop-in replacement for `synthesizeMovement`, with the same signature plus
 * the model handle. Async because ONNX inference is async.
 *
 * @param model   Loaded flow model.
 * @param start   Starting cursor position (viewport coordinates).
 * @param target  Target bounding box.
 * @param profile Behavior profile; only `precision` and `tremor` apply here —
 *                `speed` is ignored because duration is sampled from the model.
 * @param seed    RNG seed. Omit for a random gesture, pass for reproducibility.
 */
export declare function synthesizeMovementFlow(model: FlowModel, start: Point2D, target: TargetBox, profile?: Partial<BehaviorProfile>, seed?: number): Promise<TrajectoryPlan>;
/**
 * Convenience: flow movement plus post-arrival settle and click hold.
 *
 * Mirrors `synthesizeClick` for the parametric path.
 */
export declare function synthesizeClickFlow(model: FlowModel, start: Point2D, target: TargetBox, profile?: Partial<BehaviorProfile>, seed?: number): Promise<{
    trajectory: TrajectoryPlan;
    settleMs: number;
    holdMs: number;
}>;
//# sourceMappingURL=flow-synthesis.d.ts.map