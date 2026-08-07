/**
 * Flow-based pointer synthesis.
 *
 * Replaces the parametric Bezier + Fitts + overshoot pipeline with a sample
 * from a conditional normalizing flow trained on captured human gestures (see
 * flow/train_flow.py). The flow models
 *
 *     p( 79 free du, 79 free dv, log(duration_ms)
 *        | log(distance), log(target_size), terminal_stop )
 *
 * in a canonical target-relative frame: start at the origin, endpoint on the
 * +X axis at unit distance, 81 points on a uniform time grid.
 *
 * Geometry, velocity, and duration are learned jointly rather than being tuned
 * one constant at a time. Current claims come only from the session-held-out
 * audit; historical hard-coded comparisons are not runtime guarantees.
 *
 * Samples are not rejected for resembling a synthetic generator. A genuine
 * human distribution includes straight, slow, corrective, and otherwise
 * automation-like tails; scorer-guided redraws would censor those tails and
 * make the shipped distribution cleaner than the source population.
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
    schemaVersion: 2;
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
        droppedDeltaIndex: number;
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
    postprocess: {
        terminalStopProbability: number;
        terminalStopEpsilon: number;
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
 * Prevent retained long-duration human records from allocating an unbounded
 * output array. The elapsed duration is preserved; only the dispatch sampling
 * interval becomes adaptive above this limit.
 */
export declare const MAX_DISPATCH_FRAMES = 4096;
/**
 * Resample a canonical-frame path onto a fixed frame interval.
 *
 * The flow emits 81 points on its own uniform grid; dispatch needs samples at
 * the frame rate the pointer actually reports at.
 */
export declare function resampleToFrameRate(u: Float64Array, v: Float64Array, t: Float64Array, frameMs: number): {
    u: Float64Array;
    v: Float64Array;
    t: Float64Array;
};
/**
 * Turn one standardized model vector into a canonical-frame gesture.
 *
 * Mirrors `reconstruct()` in flow/audit_eval.py. Exported so the audit and the
 * runtime can be checked against each other.
 */
export declare function reconstructGesture(xStandardized: Float32Array, stats: FlowStats, forceTerminalStop?: boolean): {
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
 * @param profile Behavior profile. Flow samples the human duration distribution,
 *                then applies the persisted persona's bounded speed multiplier.
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