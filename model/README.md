# Soma model pipeline

This directory contains the reproducible local scorer pipeline. The model is a small MLP with 10,185 parameters:

```text
12 → 104 → 64 → 32 → 1
     ReLU  ReLU  ReLU Sigmoid
```

The output is an internal bot-probability benchmark, not a third-party detector score.

## Data

Human pointer paths come from Soma capture exports. Trusted v2 records are accepted directly; v1 envelopes and migrated per-record v1 data require the explicit `--allow-legacy` operator attestation and are reduced to their final contiguous active pointer burst to remove legacy idle gaps.

Synthetic negatives include:

- straight-line paths;
- constant-velocity paths;
- white-noise jitter;
- naïve smooth easing;
- random waypoint paths.

Negative durations are sampled log-uniformly across the recorder's accepted 50–10,000 ms range; long constant-velocity paths are not left as an uncovered regime.

Human samples are grouped by capture session. A small training-only Soma calibration cohort is labeled human-like to keep the bundled scorer compatible with the library's own synthesis; it is excluded from validation/test metrics. Synthetic samples are grouped by generator family and deterministic batch. Training, validation, and test groups are disjoint. The test cohort is not used for checkpoint selection.

## Train

```bash
bun run train:model -- \
  /path/to/soma-capture.json \
  --allow-legacy \
  --epochs 100 \
  --seed 42 \
  --out model/model.json \
  --metrics model/model.metrics.json
```

The command rejects invalid exports, un-attested legacy records, empty/one-class corpora, insufficient independent groups, and non-finite CLI values. Omit `--allow-legacy` for a purely v2 corpus. Output is written only after both the Soma self-consistency gate and the long-straight-negative gate pass. Metrics contain artifact basenames rather than local paths and include both gates, the canonical weight hash, feature schema, capture provenance, cohort sizes, validation/test confusion counts, and explicit limitations.

The current benchmark is intentionally scoped: one operator, known synthetic families, and no third-party detector. Do not interpret a high test accuracy as evidence of population-level authenticity.

The 12 model inputs are derived only from trajectory geometry and timing. Target-box metadata is intentionally excluded so training, promotion, and path-only runtime scoring use identical information.

## JSON loading

JSON is a strict, explicit model format. Runtime callers must load it:

```typescript
import { loadJSONModel, measureAndScore } from 'soma';

if (!await loadJSONModel('./model/model.json')) throw new Error('Invalid model');
const result = await measureAndScore(path);
```

Browser callers can use `loadJSONModelText()` with text fetched through the browser. The loader requires `soma.motion-model.v1` with `soma.motion-features.v2`; incompatible feature schemas, invalid dimensions, missing fields, `NaN`, and `Infinity` are rejected.

## ONNX export

The exporter creates a real ONNX graph containing `MatMul`, `Add`, `Relu`, and `Sigmoid` nodes:

```bash
bun run export:model -- --input model/model.json --out model/model.onnx
```

Weights are transposed into ONNX matrix layout. The graph uses input `input` with shape `[1, 12]`, output `output` with shape `[1, 1]`, and `soma.motion-features.v2` metadata. JSON and ONNX outputs are expected to agree within float32 tolerance. `loadONNXModel()` validates metadata, names, and shapes before enabling dispatch; native execution remains optional through the `onnxruntime-node` peer dependency.

## Files

- `src/dataset.ts` — capture validation, legacy migration, deterministic negative generation, session groups.
- `src/features.ts` — re-export of the canonical runtime feature extractor.
- `src/train.ts` — seeded Adam training, group-disjoint cohorts, metrics and promotion gate.
- `src/export_onnx.ts` — real protobuf graph serializer.
- `model.json` — canonical JSON weights.
- `model.onnx` — generated ONNX artifact.
- `model.metrics.json` — benchmark metadata and limitations.
