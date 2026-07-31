# Soma

Soma is a TypeScript library for deterministic behavioral input plans, measurement, local scoring, and privacy-preserving capture. It is grounded in the behavioral contracts recorded in [`clandestine-grimoire`](https://github.com/marcxavier/clandestine-grimoire).

Soma does **not** claim to bypass cside, DataDome, or any other third-party detector. The bundled scorer is an internal benchmark over the training corpus and known synthetic generators; it is not evidence of detector passage.

## What ships

- Pointer plans: target-box endpoint sampling, cubic Bezier geometry, directed overshoot/correction, Fitts-derived duration, fixed 60 Hz timing, autocorrelated tremor.
- Keystroke plans: digraph-dependent flight timing, Gaussian holds, physical US keyboard codes, Shift metadata, and explicit correction pauses.
- Scroll plans: exact distances with peak-bounded inertial lobes for long gestures.
- Measurement: curvature, acceleration reversals, velocity profile, bounded multi-lobe lognormal approximation, tremor-band energy, residual autocorrelation, and direct extraction from privacy-safe capture records.
- Dispatch helpers: absolute-time pointer, key, and wheel scheduling plus `HumanPointer.click()` with a live hit-test immediately before pointer-down.
- Scoring: explicit JSON or ONNX model loading, strict model validation, a path-only 12-feature schema, and a documented heuristic fallback.
- Capture extension: credential-field exclusion, privacy-safe digraph classes, normalized wheel units, trusted/untrusted event provenance, best-known DOM action-ready latency, collision-safe IndexedDB records, and per-record v1/v2 provenance.

## Install and build

```bash
bun install
bun run build
bun test
```

The extension is a separate unpacked Chrome artifact:

```bash
bun run build:ext
# Load ext/ at chrome://extensions with Developer mode enabled.
```

## Pointer plan and dispatch

Plans are pure data. Dispatch through your automation framework or use the generic adapters:

```typescript
import {
  HumanPointer,
  type PointerClickDriver,
} from 'soma';

const driver: PointerClickDriver = {
  async move(x, y) { await page.mouse.move(x, y); },
  async isInteractiveAt(x, y) {
    return page.evaluate(({ x, y }) => {
      const element = document.elementFromPoint(x, y);
      return !!element?.closest('a,button,input,select,textarea,[role="button"],[role="link"]');
    }, { x, y });
  },
  async down() { await page.mouse.down(); },
  async up() { await page.mouse.up(); },
};

const pointer = new HumanPointer(driver, { x: 100, y: 200 });
await pointer.click({ x: 500, y: 400, width: 120, height: 40 }, 42);
```

`HumanPointer.click()` waits for the plan's pre-move dwell, dispatches each point at its absolute `tMs`, waits for post-arrival settle, performs `isInteractiveAt()` against the live page, and refuses the click if the endpoint is no longer interactive.

For lower-level drivers, `dispatchTrajectory`, `dispatchKeystrokePlan`, and `dispatchScrollPlan` accept an injectable delay function for deterministic tests.

## Keystrokes

```typescript
import { dispatchKeystrokePlan, synthesizeKeystrokes } from 'soma';

const plan = synthesizeKeystrokes('Hello!?', { mistakeRate: 0.02 }, 42);
await dispatchKeystrokePlan(plan, {
  down: (event) => page.keyboard.down(event.key),
  up: (event) => page.keyboard.up(event.key),
});
```

`KeyEvent.downMs` is an absolute offset. The dispatcher schedules keydown and keyup on one timeline, so overlapping holds are not accidentally serialized. Uppercase and shifted punctuation expose `shiftKey` and `modifiers` metadata; the consumer is responsible for sending the corresponding modifier through its trusted input channel.

## Scroll

```typescript
import { dispatchScrollPlan, synthesizeScroll } from 'soma';

await dispatchScrollPlan(synthesizeScroll(500, undefined, 42), {
  wheel: (deltaY) => page.mouse.wheel({ deltaY }),
});
```

Short targets sum exactly to the requested distance. Long distances use repeated peak-bounded inertial lobes instead of an unbounded residual frame. Device-specific wheel normalization belongs at the browser adapter/capture boundary.

## Measurement and scoring

```typescript
import { measureAndScore } from 'soma';

const result = await measureAndScore(path);
console.log(result.velocity_profile, result.sigma_lognormal_fit);
console.log(result.neural_bot_score); // local score: 0 human-like, 1 bot-like
```

No model is loaded implicitly. The heuristic is used until a model is explicitly loaded:

```typescript
import { loadJSONModel } from 'soma';

if (!await loadJSONModel('./model/model.json')) {
  throw new Error('Invalid model weights');
}
```

`loadJSONModelText()` is available for browser callers that fetch the JSON themselves. JSON loading requires `soma.motion-model.v1` with `soma.motion-features.v2`, so weights trained for a different feature meaning cannot load silently. `loadONNXModel()` is available when the optional `onnxruntime-node` peer dependency is installed. The repository's `model/model.onnx` is generated by the protobuf exporter and parity-checked against JSON during release verification.

The scorer's feature vector is path-only: optional target arguments remain accepted for source compatibility but do not change inference. This prevents training or promotion checks from learning whether target metadata happened to be available.

## Training

The included extension exports v2 captures. v1 envelopes and migrated per-record v1 data require explicit operator attestation through `--allow-legacy`; their paths are reduced to the final contiguous active burst because the old recorder included idle gaps.

```bash
bun run train:model -- \
  /path/to/soma-capture-2026-08-01T04-41-10.json \
  --allow-legacy \
  --epochs 100 \
  --seed 42 \
  --out model/model.json \
  --metrics model/model.metrics.json
```

The command uses full-duration deterministic negatives, session/group-disjoint train/validation/test cohorts, a training-only set of Soma compatibility paths, strict class checks, a Soma self-consistency gate, and a long-straight-negative gate. The metrics file records both promotion gates, the model hash, feature schema, cohort sizes, confusion counts, and limitations. It does not establish population-level authenticity or third-party detector performance.

## Capture privacy and coverage

The extension stores data locally in IndexedDB and never sends it to a network endpoint. v2 capture excludes password, credential, token, authentication, and username fields—including fields exposed through an open shadow path—and stores no printable `key`, physical `code`, or field value. Ctrl/Meta/Alt shortcuts are excluded from printable timing. The recorder stores hand/key-kind/digraph classes, trusted commit counts and `inputType`, normalized wheel data, and a separate provenance stream containing trusted and untrusted non-sensitive DOM events. Only trusted v2 movement records enter training by default; migrated v1 records require explicit attestation.

Action latency uses the best monotonic time at which the target element was observed inserted or enabled; controls present from navigation are anchored at navigation start. Recorder interaction counts exclude the final consequential action by default, or accept an explicit action timestamp. This is a recorder estimate, not an application-declared readiness timestamp. Capture covers regular and related opaque all-frame DOM pointer, keyboard, wheel, and actionable-event streams while excluding credential-field activity; opaque custom-element targets whose closed shadow contents cannot be inspected are suppressed conservatively. Touch/stylus trajectories, scrollbar and keyboard scrolling, page-destroyed pending sends, browser chrome/omnibox activity, and application semantics unavailable to a content script remain outside its boundary. Treat exports as a behavioral corpus, not a complete browsing history.

## Project status

The core contracts are regression-tested with Bun. The local model benchmark uses one operator's capture and known synthetic generators. External detector claims remain intentionally unmade until independently measured against versioned fixtures and a declared test protocol.

MIT licensed; see `LICENSE`.
