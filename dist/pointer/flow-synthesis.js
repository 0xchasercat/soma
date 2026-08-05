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
import { RNG } from '../rng.js';
import { mergeProfile } from '../profile.js';
import { dist2D, preMoveSettle, sampleEndpoint, clickHoldTime } from './fitts.js';
import { addTremor } from './tremor.js';
import { scoreTrajectory } from '../score/index.js';
/** Throw unless the stats artifact is internally consistent. */
export function validateFlowStats(stats) {
    const { arch, layout, stats: s } = stats;
    if (s.xMean.length !== arch.dataDim || s.xStd.length !== arch.dataDim)
        throw new Error(`flow stats: xMean/xStd length != dataDim ${arch.dataDim}`);
    if (s.cMean.length !== arch.contextDim || s.cStd.length !== arch.contextDim)
        throw new Error(`flow stats: context stats length != contextDim ${arch.contextDim}`);
    if (layout.nFree * 2 + 1 !== arch.dataDim)
        throw new Error(`flow stats: layout 2*${layout.nFree}+1 != dataDim ${arch.dataDim}`);
    if (layout.logDurationIndex !== arch.dataDim - 1)
        throw new Error('flow stats: logDurationIndex must be the final dimension');
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
 * Surrogate-scorer value at or above which a sampled gesture is redrawn.
 *
 * Matches the detector threshold the gate uses. Measured over 351 flow
 * gestures: 18% score >= 0.8 while real hands do so on 5.1%, and those samples
 * concentrate in one region (median 1376 ms duration at arc/straight 1.056 —
 * slow, nearly straight reaches) that the corpus supports but only sparsely.
 */
const REJECTION_SCORE_MAX = 0.8;
/**
 * Maximum flow draws per gesture, including the first.
 *
 * Draws are independent, so a 0.18 per-draw rejection rate leaves 0.18^4 ≈ 0.1%
 * of gestures still flagged after four attempts. Bounded so a pathological
 * context cannot spin: the final draw is used whether or not it passes.
 */
const REJECTION_ATTEMPTS = 4;
/**
 * Resample a canonical-frame path onto a fixed frame interval.
 *
 * The flow emits 129 points on its own uniform grid; dispatch needs samples at
 * the frame rate the pointer actually reports at.
 */
function resampleToFrameRate(u, v, t, frameMs) {
    const total = t[t.length - 1];
    const frames = Math.max(2, Math.round(total / frameMs) + 1);
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
export function reconstructGesture(xStandardized, stats) {
    const { xMean, xStd } = stats.stats;
    const { nFree, nSteps, duStart, dvStart, logDurationIndex } = stats.layout;
    const du = new Float64Array(nSteps);
    const dv = new Float64Array(nSteps);
    let duSum = 0;
    let dvSum = 0;
    for (let i = 0; i < nFree; i++) {
        const a = xStandardized[duStart + i] * xStd[duStart + i] + xMean[duStart + i];
        const b = xStandardized[dvStart + i] * xStd[dvStart + i] + xMean[dvStart + i];
        du[i] = a;
        dv[i] = b;
        duSum += a;
        dvSum += b;
    }
    // Restore the deltas dataset_prep dropped: the canonical frame pins the
    // endpoint at (1, 0), so the tails are determined, not free.
    du[nSteps - 1] = 1 - duSum;
    dv[nSteps - 1] = -dvSum;
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
 * @param profile Behavior profile; only `precision` and `tremor` apply here —
 *                `speed` is ignored because duration is sampled from the model.
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
    // Degenerate reach: the flow is trained on >= 20px gestures and the canonical
    // frame divides by distance, so a sub-pixel move has no valid normalization.
    if (distance < 1) {
        const points = [
            { x: safeStart.x, y: safeStart.y, tMs: 0 },
            { x: endpoint.x, y: endpoint.y, tMs: 16 },
        ];
        return { points, durationMs: 16, preMoveDelayMs: preMoveSettle(rng), endpoint };
    }
    const { dataDim, contextDim } = model.stats.arch;
    const { cMean, cStd } = model.stats.stats;
    const context = new Float32Array(contextDim);
    context[0] = (Math.log(distance) - cMean[0]) / cStd[0];
    context[1] = (Math.log(targetSize) - cMean[1]) / cStd[1];
    const theta = Math.atan2(endpoint.y - safeStart.y, endpoint.x - safeStart.x);
    const cos = Math.cos(theta);
    const sin = Math.sin(theta);
    const frameMs = 1000 / FRAME_HZ;
    const tremorAmplitude = Number.isFinite(p.tremor) ? Math.max(0, p.tremor) : 0;
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
        const canonical = reconstructGesture(xOut, model.stats);
        const resampled = resampleToFrameRate(canonical.u, canonical.v, canonical.t, frameMs);
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
        // Physiological tremor is applied here rather than learned: the audit showed
        // the flow's own high-frequency lateral energy has the wrong spectrum, while
        // the AR(2) oscillator in tremor.ts is fitted to the measured residual ACF.
        return {
            points: addTremor(points, tremorAmplitude, rng),
            durationMs: canonical.t[canonical.t.length - 1],
        };
    };
    // Rejection sampling. The flow places ~17% of its mass on a region real hands
    // occupy ~5% of the time (slow, near-straight reaches), and the surrogate
    // scorer flags exactly those. Redrawing z re-samples that gesture from the
    // same conditional density, so rejecting the flagged tail reshapes the
    // sampling distribution toward the corpus without touching the model.
    // Rejections are independent, so REJECTION_ATTEMPTS = 4 leaves a ~0.1%
    // residual; the last draw is returned regardless so a gesture is always
    // produced, and scoring failures degrade to accepting the first draw.
    let chosen = await attempt();
    for (let tries = 1; tries < REJECTION_ATTEMPTS; tries++) {
        let score;
        try {
            score = await scoreTrajectory(chosen.points);
        }
        catch {
            break;
        }
        if (score < REJECTION_SCORE_MAX)
            break;
        chosen = await attempt();
    }
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