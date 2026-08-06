# Phase 4 — Flow Integration Status

> Historical design record (superseded). Runtime integration is complete in
> `src/pointer/flow-synthesis.ts`; the bundled `model/flow.onnx` and
> `model/flow.stats.json` use a 159-dimensional interface. Production callers
> now require successful ONNX initialization and persist artifact hashes. The
> blocked-export discussion below documents the earlier 255-dimensional model.

## Delivered (Phases 1–3)

✅ **Dataset preparation** — `flow_dataset.pt` with 19,755 gestures (255-dim, non-degenerate)  
✅ **Training** — `flow.pt` with val NLL −686.2 (−2.69/dim), converged cleanly  
✅ **Audit** — Headline discriminator features dropped 57–85 points vs. parametric baseline

## Phase 4 — Runtime Integration

### Blocking Issue: ONNX Export

PyTorch 2.5's new exporter (`torch.export`) cannot trace through nflows' RQ-spline conditionals:

```
torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode: 
Could not guard on data-dependent expression Eq(u1, 1)
  File "nflows/transforms/splines/rational_quadratic.py", line 42
    if torch.any(inside_interval_mask):
```

The issue is that nflows checks whether spline inputs lie inside the bounded region at runtime, and the new tracer can't symbolically resolve that boolean. The old JIT tracer could handle this, but it's deprecated.

**Workarounds (in priority order):**

1. **Patch nflows locally** — Replace `if torch.any(mask):` with `torch.where(mask, spline_path, identity_path)` to make the conditional traceable. This is a 5-line fix to `/opt/homebrew/lib/python3.14/site-packages/nflows/transforms/splines/rational_quadratic.py:42`.

2. **Use `torch.jit.trace`** (deprecated but functional) — The legacy JIT tracer executes the conditional on concrete inputs and bakes the taken branch into the graph. For a model that always stays in-bounds (ours does, tail_bound=6 and standardized data), this works.

3. **Call Python from TypeScript** — Spawn a Python subprocess with the model loaded, send (z, c) over stdin, receive x over stdout. Deterministic but adds 50–100ms latency per gesture.

### Recommended Path Forward

**Short-term (MVP):** Option 3 — Python subprocess. Proves deterministic replay and validates the full pipeline without fixing the export.

**Production:** Option 1 — Patch nflows and re-export. The patch is trivial and doesn't change the math, just makes the code export-friendly.

### TypeScript Integration Architecture (once ONNX works)

```typescript
// src/pointer/flow-synthesis.ts

import * as ort from 'onnxruntime-node';
import { RNG } from '../rng.js';
import type { Point2D, TargetBox, BehaviorProfile, TrajectoryPlan } from '../types.js';

let flowSession: ort.InferenceSession | null = null;
let flowStats: { x_mean: Float32Array; x_std: Float32Array; c_mean: Float32Array; c_std: Float32Array } | null = null;

export async function loadFlowModel(modelPath: string, statsPath: string): Promise<void> {
  flowSession = await ort.InferenceSession.create(modelPath);
  const statsBuffer = await Bun.file(statsPath).arrayBuffer();
  // ... deserialize stats from the .pt checkpoint metadata
}

export function synthesizeMovementFlow(
  start: Point2D,
  target: TargetBox,
  profile?: Partial<BehaviorProfile>,
  seed?: number,
): TrajectoryPlan {
  if (!flowSession || !flowStats) {
    throw new Error('Flow model not loaded');
  }

  const rng = new RNG(seed);
  const precision = profile?.precision ?? 0.75;
  const endpoint = sampleEndpoint(target, precision, rng);
  const distance = dist2D(start, endpoint);
  const targetSize = Math.min(target.width, target.height);

  // 1. Sample z ~ N(0, I) from the seeded RNG
  const z = new Float32Array(255);
  for (let i = 0; i < 255; i++) {
    z[i] = rng.nextGaussian(0, 1);
  }

  // 2. Standardize context
  const c_raw = new Float32Array([Math.log(distance), Math.log(targetSize)]);
  const c = new Float32Array(2);
  for (let i = 0; i < 2; i++) {
    c[i] = (c_raw[i] - flowStats.c_mean[i]) / flowStats.c_std[i];
  }

  // 3. Run ONNX inverse: (z, c) → x_standardized
  const feeds = {
    z: new ort.Tensor('float32', z, [1, 255]),
    context: new ort.Tensor('float32', c, [1, 2]),
  };
  const results = await flowSession.run(feeds);
  const x_std = results.x.data as Float32Array;

  // 4. Denormalize: x_std → x_raw
  const x_raw = new Float32Array(255);
  for (let i = 0; i < 255; i++) {
    x_raw[i] = x_std[i] * flowStats.x_std[i] + flowStats.x_mean[i];
  }

  // 5. Reconstruct gesture in target-relative frame
  const du = new Float32Array(128);
  const dv = new Float32Array(128);
  for (let i = 0; i < 127; i++) {
    du[i] = x_raw[i];
    dv[i] = x_raw[127 + i];
  }
  // Restore the dropped constrained deltas
  du[127] = 1.0 - du.slice(0, 127).reduce((a, b) => a + b, 0);
  dv[127] = 0.0 - dv.slice(0, 127).reduce((a, b) => a + b, 0);

  const duration_ms = Math.exp(x_raw[254]);

  // 6. Integrate deltas → path in normalized frame
  const u = new Float32Array(129);
  const v = new Float32Array(129);
  u[0] = 0; v[0] = 0;
  for (let i = 0; i < 128; i++) {
    u[i + 1] = u[i] + du[i];
    v[i + 1] = v[i] + dv[i];
  }

  // 7. Denormalize: scale by distance, rotate, translate
  const theta = Math.atan2(endpoint.y - start.y, endpoint.x - start.x);
  const cos_t = Math.cos(theta);
  const sin_t = Math.sin(theta);

  const points: TrajectoryPoint[] = [];
  for (let i = 0; i < 129; i++) {
    const u_px = u[i] * distance;
    const v_px = v[i] * distance;
    const x = start.x + cos_t * u_px - sin_t * v_px;
    const y = start.y + sin_t * u_px + cos_t * v_px;
    const tMs = (i / 128) * duration_ms;
    points.push({ x, y, tMs });
  }

  // 8. Add physiological tremor (reuse existing addTremor from tremor.ts)
  const tremorAmp = profile?.tremor ?? 0.3;
  const finalPath = addTremor(points, tremorAmp, rng);

  return {
    points: finalPath,
    durationMs: duration_ms,
    preMoveDelayMs: preMoveSettle(rng),
    endpoint,
  };
}
```

### Deterministic Replay Verification

The key property: **identical seed → identical trajectory**.

Test:
```typescript
const seed = 12345;
const start = { x: 100, y: 200 };
const target = { x: 400, y: 300, width: 50, height: 50 };

const traj1 = synthesizeMovementFlow(start, target, {}, seed);
const traj2 = synthesizeMovementFlow(start, target, {}, seed);

assert(traj1.points.length === traj2.points.length);
for (let i = 0; i < traj1.points.length; i++) {
  assert(traj1.points[i].x === traj2.points[i].x);
  assert(traj1.points[i].y === traj2.points[i].y);
  assert(traj1.points[i].tMs === traj2.points[i].tMs);
}
```

This holds because:
1. `RNG` is deterministic (mulberry32 + Box-Muller)
2. ONNX model execution is deterministic (no dropout, no randomness)
3. The denormalization math is pure arithmetic

### Next Steps

1. **Patch nflows** — 5-line fix to make the RQ-spline traceable
2. **Re-export ONNX** — Run the patched export
3. **Write `flow-synthesis.ts`** — Implement the above
4. **Add to Soma exports** — `export { synthesizeMovementFlow, loadFlowModel } from './pointer/flow-synthesis.js'`
5. **Integration test** — Verify deterministic replay in `test/behavior.test.ts`
6. **Update sip-eval** — Wire flow synthesis as an optional mode: `SIP_SYNTHESIS_MODE=flow|bezier`

### Known Limitations (from Phase 3 audit)

The current model has excellent **shape** (overshoot, peak timing, bell-profile all near-perfect) but broken **microstructure**:

- **Submovement count**: 100% separable (40 vs. 13) — velocity profile too fragmented
- **Tremor energy**: 97.9% separable — high-freq lateral jerk concentration wrong
- **Terminal deceleration**: 99.8% separable — lands at 72% peak speed vs. 0.4%
- **Lateral deviation**: 55.7% separable — 3.4% vs. 23.5%

These are fixable (higher `N_BINS`, separate duration head, explicit tremor injection) but out of scope for the initial proof-of-concept.

## Deliverables

**Completed:**
- ✅ Phase 1: Dataset with 19,755 clean gestures
- ✅ Phase 2: Trained flow model (−686 NLL, no NaN, converged)
- ✅ Phase 3: Audit proving 57–85 pt improvement on headline features
- ✅ Integration architecture documented above

**Blocked on ONNX export fix:**
- ⏸️ ONNX model file
- ⏸️ TypeScript `flow-synthesis.ts`
- ⏸️ Deterministic replay test

**Estimated fix time:** 1–2 hours (patch nflows, re-export, write TypeScript, test)
