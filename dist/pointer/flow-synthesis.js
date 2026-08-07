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
import { RNG } from '../rng.js';
import { mergeProfile } from '../profile.js';
import { dist2D, preMoveSettle, sampleEndpoint, clickHoldTime } from './fitts.js';
/** Throw unless the stats artifact is internally consistent. */
export function validateFlowStats(stats) {
    const { arch, layout, stats: s } = stats;
    if (stats.schemaVersion !== 2)
        throw new Error(`flow stats: unsupported schemaVersion ${String(stats.schemaVersion)}; expected 2`);
    if (arch.contextDim !== 3)
        throw new Error(`flow stats: contextDim ${arch.contextDim} != schema-v2 contextDim 3`);
    if (s.xMean.length !== arch.dataDim || s.xStd.length !== arch.dataDim)
        throw new Error(`flow stats: xMean/xStd length != dataDim ${arch.dataDim}`);
    if (s.cMean.length !== arch.contextDim || s.cStd.length !== arch.contextDim)
        throw new Error(`flow stats: context stats length != contextDim ${arch.contextDim}`);
    if (layout.nFree * 2 + 1 !== arch.dataDim)
        throw new Error(`flow stats: layout 2*${layout.nFree}+1 != dataDim ${arch.dataDim}`);
    if (layout.logDurationIndex !== arch.dataDim - 1)
        throw new Error('flow stats: logDurationIndex must be the final dimension');
    if (!Number.isInteger(layout.droppedDeltaIndex) ||
        layout.droppedDeltaIndex < 0 || layout.droppedDeltaIndex >= layout.nSteps)
        throw new Error('flow stats: droppedDeltaIndex must address a spatial delta');
    if (!(stats.postprocess?.terminalStopProbability >= 0 &&
        stats.postprocess.terminalStopProbability <= 1))
        throw new Error('flow stats: terminalStopProbability must be in [0, 1]');
    if (!(stats.postprocess.terminalStopEpsilon > 0))
        throw new Error('flow stats: terminalStopEpsilon must be positive');
    for (const v of s.xStd)
        if (!(v > 0))
            throw new Error('flow stats: xStd must be strictly positive');
    for (const v of s.cStd)
        if (!(v > 0))
            throw new Error('flow stats: cStd must be strictly positive');
}
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
export async function loadFlowModel(options) {
    const stats = JSON.parse(options.statsJson);
    validateFlowStats(stats);
    const session = await options.createSession(options.onnxPath);
    return { session, stats, tensor: options.tensor };
}
/** Output frame rate; matches the parametric path and typical mouse reporting. */
const FRAME_HZ = 60;
/**
 * Prevent retained long-duration human records from allocating an unbounded
 * output array. The elapsed duration is preserved; only the dispatch sampling
 * interval becomes adaptive above this limit.
 */
export const MAX_DISPATCH_FRAMES = 4096;
/**
 * Resample a canonical-frame path onto a fixed frame interval.
 *
 * The flow emits 81 points on its own uniform grid; dispatch needs samples at
 * the frame rate the pointer actually reports at.
 */
export function resampleToFrameRate(u, v, t, frameMs) {
    const total = t[t.length - 1];
    const desiredFrames = Math.round(total / frameMs) + 1;
    const frames = Math.max(2, Math.min(MAX_DISPATCH_FRAMES, desiredFrames));
    const ru = new Float64Array(frames);
    const rv = new Float64Array(frames);
    const rt = new Float64Array(frames);
    let src = 0;
    for (let i = 0; i < frames; i++) {
        const target = i === frames - 1 ? total : (i * total) / (frames - 1);
        while (src < t.length - 2 && t[src + 1] < target)
            src++;
        const t0 = t[src];
        const t1 = t[src + 1];
        const span = t1 - t0;
        const w = span > 1e-9 ? (target - t0) / span : 0;
        ru[i] = u[src] + (u[src + 1] - u[src]) * w;
        rv[i] = v[src] + (v[src + 1] - v[src]) * w;
        rt[i] = target;
    }
    return { u: ru, v: rv, t: rt };
}
/**
 * Turn one standardized model vector into a canonical-frame gesture.
 *
 * Mirrors `reconstruct()` in flow/audit_eval.py. Exported so the audit and the
 * runtime can be checked against each other.
 */
export function reconstructGesture(xStandardized, stats, forceTerminalStop = false) {
    const { xMean, xStd } = stats.stats;
    const { nFree, nSteps, droppedDeltaIndex, duStart, dvStart, logDurationIndex } = stats.layout;
    const du = new Float64Array(nSteps);
    const dv = new Float64Array(nSteps);
    let duSum = 0;
    let dvSum = 0;
    let freeIndex = 0;
    for (let step = 0; step < nSteps; step++) {
        if (step === droppedDeltaIndex)
            continue;
        const a = xStandardized[duStart + freeIndex] * xStd[duStart + freeIndex] + xMean[duStart + freeIndex];
        const b = xStandardized[dvStart + freeIndex] * xStd[dvStart + freeIndex] + xMean[dvStart + freeIndex];
        const modeledU = forceTerminalStop && step === nSteps - 1 ? 0 : a;
        const modeledV = forceTerminalStop && step === nSteps - 1 ? 0 : b;
        du[step] = modeledU;
        dv[step] = modeledV;
        duSum += modeledU;
        dvSum += modeledV;
        freeIndex++;
    }
    // Restore the omitted middle deltas. The canonical endpoint constraints make
    // them determined while leaving the measured touchdown deltas explicit.
    du[droppedDeltaIndex] = 1 - duSum;
    dv[droppedDeltaIndex] = -dvSum;
    const durationMs = Math.exp(xStandardized[logDurationIndex] * xStd[logDurationIndex] + xMean[logDurationIndex]);
    const u = new Float64Array(nSteps + 1);
    const v = new Float64Array(nSteps + 1);
    for (let i = 0; i < nSteps; i++) {
        u[i + 1] = u[i] + du[i];
        v[i + 1] = v[i] + dv[i];
    }
    const t = new Float64Array(nSteps + 1);
    for (let i = 0; i <= nSteps; i++)
        t[i] = (i / nSteps) * durationMs;
    return { u, v, t, durationMs };
}
// ─── Synthesis ────────────────────────────────────────────────────────────────
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
export async function synthesizeMovementFlow(model, start, target, profile, seed) {
    const finite = (value, fallback) => Number.isFinite(value) ? value : fallback;
    const safeStart = { x: finite(start.x, 0), y: finite(start.y, 0) };
    const safeTarget = {
        x: finite(target.x, 0),
        y: finite(target.y, 0),
        width: Math.max(1, finite(target.width, 1)),
        height: Math.max(1, finite(target.height, 1)),
    };
    const p = mergeProfile(profile);
    const rng = new RNG(seed);
    const precision = Number.isFinite(p.precision)
        ? Math.max(0, Math.min(1, p.precision))
        : 0.75;
    const endpoint = sampleEndpoint(safeTarget, precision, rng);
    const distance = dist2D(safeStart, endpoint);
    const targetSize = Math.max(1, Math.min(safeTarget.width, safeTarget.height));
    // Only a zero-displacement reach is non-canonical. Short human corrections
    // remain in training and must not be routed away from the learned density.
    if (distance <= 1e-6) {
        const points = [
            { x: safeStart.x, y: safeStart.y, tMs: 0 },
            { x: endpoint.x, y: endpoint.y, tMs: 16 },
        ];
        return { points, durationMs: 16, preMoveDelayMs: preMoveSettle(rng), endpoint };
    }
    const { dataDim, contextDim } = model.stats.arch;
    const { cMean, cStd } = model.stats.stats;
    // Exact stationary touchdown is a discrete human state, not a value a
    // continuous density can represent. Sample it first and condition the flow
    // on the selected regime so moving-touchdown samples learn their own tail.
    const forceTerminalStop = rng.next() < model.stats.postprocess.terminalStopProbability;
    const context = new Float32Array(contextDim);
    context[0] = (Math.log(distance) - cMean[0]) / cStd[0];
    context[1] = (Math.log(targetSize) - cMean[1]) / cStd[1];
    context[2] = ((forceTerminalStop ? 1 : 0) - cMean[2]) / cStd[2];
    const theta = Math.atan2(endpoint.y - safeStart.y, endpoint.x - safeStart.x);
    const cos = Math.cos(theta);
    const sin = Math.sin(theta);
    const frameMs = 1000 / FRAME_HZ;
    const durationScale = 1 / Math.max(0.2, Math.min(3, p.speed));
    /** One draw from the flow, mapped into the viewport with tremor applied. */
    const attempt = async () => {
        // z ~ N(0, I) from the seeded RNG: this is the only stochastic input, so
        // determinism of the whole plan reduces to determinism of the RNG.
        const z = new Float32Array(dataDim);
        for (let i = 0; i < dataDim; i++)
            z[i] = rng.nextGaussian(0, 1);
        const outputs = await model.session.run({
            z: model.tensor(z, [1, dataDim]),
            context: model.tensor(context, [1, contextDim]),
        });
        const xOut = outputs.x?.data;
        if (!xOut || xOut.length < dataDim) {
            throw new Error('flow inference returned no usable output tensor');
        }
        for (let i = 0; i < dataDim; i++) {
            if (!Number.isFinite(xOut[i])) {
                throw new Error(`flow inference produced non-finite value at index ${i}`);
            }
        }
        const canonical = reconstructGesture(xOut, model.stats, forceTerminalStop);
        const personaTime = Float64Array.from(canonical.t, (value) => value * durationScale);
        const resampled = resampleToFrameRate(canonical.u, canonical.v, personaTime, frameMs);
        // Canonical frame -> viewport: scale by distance, rotate onto the start->end
        // direction, translate to the start point.
        const n = resampled.t.length;
        const points = new Array(n);
        for (let i = 0; i < n; i++) {
            const ux = resampled.u[i] * distance;
            const vy = resampled.v[i] * distance;
            points[i] = {
                x: safeStart.x + cos * ux - sin * vy,
                y: safeStart.y + sin * ux + cos * vy,
                tMs: resampled.t[i],
            };
        }
        // Pin the endpoints exactly: interpolation and smoothing must not move the
        // click target by a fraction of a pixel.
        points[0] = { x: safeStart.x, y: safeStart.y, tMs: 0 };
        points[n - 1] = { x: endpoint.x, y: endpoint.y, tMs: resampled.t[n - 1] };
        if (forceTerminalStop && n > 2) {
            // A canonical stop occupies only 1/80 of the duration and can disappear
            // when resampled to 60 Hz. Reserve the last dispatch interval for the
            // measured stationary mouseup state instead of losing the discrete atom.
            points[n - 2] = { x: endpoint.x, y: endpoint.y, tMs: resampled.t[n - 2] };
        }
        return {
            // The flow is trained on complete captured microstructure. Adding the
            // legacy AR(2) oscillator a second time increased held-out separability.
            points,
            durationMs: resampled.t[resampled.t.length - 1],
        };
    };
    const chosen = await attempt();
    return {
        points: chosen.points,
        durationMs: chosen.durationMs,
        preMoveDelayMs: preMoveSettle(rng),
        endpoint,
    };
}
/**
 * Convenience: flow movement plus post-arrival settle and click hold.
 *
 * Mirrors `synthesizeClick` for the parametric path.
 */
export async function synthesizeClickFlow(model, start, target, profile, seed) {
    const trajectory = await synthesizeMovementFlow(model, start, target, profile, seed);
    // Derive from a forked RNG so click timing does not depend on how many draws
    // the trajectory consumed.
    const rng = new RNG(seed).fork(0x63);
    return {
        trajectory,
        settleMs: Math.max(20, rng.nextGaussian(80, 30)),
        holdMs: clickHoldTime(rng),
    };
}
//# sourceMappingURL=flow-synthesis.js.map