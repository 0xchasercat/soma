# Conditional flow integration

The conditional pointer flow is already integrated in
`src/pointer/flow-synthesis.ts`. The canonical shipped pair is:

- `model/flow.onnx` — inverse flow, `z + context -> standardized gesture`;
- `model/flow.stats.json` — schema v2 normalization and vector layout.

Load both through `loadFlowModel`, then pass the resulting model to
`synthesizeMovementFlow`, `synthesizeClickFlow`, or `HumanPointer`.

```ts
import * as ort from 'onnxruntime-node';
import {
  FLOW_ONNX_URL,
  FLOW_STATS_URL,
  loadFlowModel,
  synthesizeMovementFlow,
} from 'soma';

const flow = await loadFlowModel({
  onnxPath: fileURLToPath(FLOW_ONNX_URL),
  statsJson: await readFile(fileURLToPath(FLOW_STATS_URL), 'utf8'),
  createSession: (path) => ort.InferenceSession.create(path) as never,
  tensor: (data, dims) => new ort.Tensor('float32', data, dims) as never,
});

const plan = await synthesizeMovementFlow(
  flow,
  { x: 10, y: 10 },
  { x: 500, y: 300, width: 80, height: 32 },
  {},
  42,
);
```

## Artifact contract

The model vector has 159 dimensions:

```text
79 on-axis deltas, excluding constrained step 40
79 lateral deltas, excluding constrained step 40
1 log-duration
```

At runtime, step 40 is recovered from the exact canonical endpoint constraint.
The final observed delta remains explicit so touchdown velocity is learned.
The context is standardized `log(distance_px), log(target_size_px),
terminal_stop`. The last value conditions the continuous flow on the sampled
branch of the measured stationary-touchdown mixture.

Stats schema v1 dropped the final delta and is intentionally rejected by the
runtime. Never pair an ONNX model with stats from a different training run.

## Regeneration and validation

```bash
CAPTURE=/path/to/soma-capture.json bun run prep:flow
bun run train:flow
bun run test:flow
bun run verify:flow
python3 flow/audit_eval.py --n 3000
bun test
bun run typecheck
```

Preparation applies no behavioral distance, duration, straightness, or
curvature filters. Validation is grouped by capture session. The audit reports
the raw flow after dispatch resampling and the actually shipped default path
after AR(2) tremor, making post-processing effects explicit.

Long-duration samples retain their elapsed duration but dispatch is capped at
4,096 points with an adaptive interval, preventing an extreme genuine pause
from allocating an unbounded frame array.
