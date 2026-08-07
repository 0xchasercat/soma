import { describe, expect, test } from 'bun:test';
import {
  HumanPointer,
  RNG,
  dispatchKeystrokePlan,
  dispatchScrollPlan,
  dispatchTrajectory,
  extractFeatureVector,
  extractCapturedInteractionFeatures,
  extractCapturedKeystrokeFeatures,
  fitSigmaLognormal,
  loadJSONModelText,
  measureTrajectory,
  measureTremor,
  mergeProfile,
  resample60Hz,
  runJSONInference,
  synthesizeKeystrokes,
  synthesizeMovement,
  synthesizeScroll,
  validateModelWeights,
} from '../src/index.js';
import type { RawInteraction, RawKeystroke } from '../src/index.js';

const approx = (a: number, b: number, tolerance = 1e-8) => Math.abs(a - b) <= tolerance;

describe('pointer synthesis', () => {
  test('emits finite target-aligned bell-shaped trajectories', () => {
    for (let seed = 0; seed < 64; seed++) {
      const plan = synthesizeMovement(
        { x: 100, y: 120 },
        { x: 640, y: 330, width: 120, height: 44 },
        undefined,
        seed,
      );
      const first = plan.points[0]!;
      const last = plan.points.at(-1)!;
      const measured = measureTrajectory(plan.points);
      expect(plan.points.every((point) => Number.isFinite(point.x) && Number.isFinite(point.y) && Number.isFinite(point.tMs))).toBe(true);
      expect(first).toEqual({ x: 100, y: 120, tMs: 0 });
      expect(last.x).toBe(plan.endpoint.x);
      expect(last.y).toBe(plan.endpoint.y);
      expect(last.x).toBeGreaterThanOrEqual(640);
      expect(last.x).toBeLessThanOrEqual(760);
      expect(last.y).toBeGreaterThanOrEqual(330);
      expect(last.y).toBeLessThanOrEqual(374);
      expect(measured.velocity_profile).toBe('bell');
      expect(measured.submovement_count).toBeGreaterThanOrEqual(1);
      expect(measured.sigma_lognormal_fit).toBeGreaterThan(0.5);
      expect(measured.has_micro_tremor).toBe(true);
      for (let i = 1; i < plan.points.length - 1; i++) {
        expect(approx(plan.points[i]!.tMs - plan.points[i - 1]!.tMs, 1000 / 60, 1e-7)).toBe(true);
      }
    }
  });

  test('sanitizes profiles and tiny targets without hanging or escaping', () => {
    const profile = mergeProfile({ speed: 0, wpm: Number.NaN, tremor: Number.POSITIVE_INFINITY });
    expect(profile.speed).toBe(0.2);
    expect(profile.wpm).toBe(65);
    expect(profile.tremor).toBe(0.3);
    const plan = synthesizeMovement(
      { x: Number.NaN, y: Number.POSITIVE_INFINITY },
      { x: 10, y: 20, width: 1, height: 1 },
      profile,
      1,
    );
    expect(plan.endpoint.x).toBeGreaterThanOrEqual(10);
    expect(plan.endpoint.x).toBeLessThanOrEqual(11);
    expect(plan.endpoint.y).toBeGreaterThanOrEqual(20);
    expect(plan.endpoint.y).toBeLessThanOrEqual(21);
  });

  test('never appends a sub-frame terminal correction', () => {
    const plan = synthesizeMovement(
      { x: 0, y: 0 },
      { x: 90, y: -10, width: 20, height: 20 },
      { precision: 1, speed: 1.0381027554944509, tremor: 0.3 },
      7,
    );
    const intervals = plan.points.slice(1).map((point, index) => point.tMs - plan.points[index]!.tMs);
    expect(Math.min(...intervals)).toBeGreaterThanOrEqual((1000 / 60) / 2);
    expect(plan.points.at(-1)!.tMs).toBeCloseTo(plan.durationMs, 10);
  });

  test('resamples on a fixed grid and preserves the final endpoint', () => {
    const path = [
      { x: 0, y: 0, tMs: 0 },
      { x: 50, y: 0, tMs: 400 },
      { x: 100, y: 0, tMs: 1000 },
    ];
    const sampled = resample60Hz(path);
    expect(sampled[0]).toEqual(path[0]);
    expect(sampled.at(-1)).toEqual(path.at(-1));
    for (let i = 1; i < sampled.length - 1; i++) {
      expect(approx(sampled[i]!.tMs - sampled[i - 1]!.tMs, 1000 / 60, 1e-7)).toBe(true);
    }
  });

  test('distinguishes a bell lobe from a flat profile', () => {
    const bell = Array.from({ length: 60 }, (_, index) => {
      const t = (index + 0.5) / 60;
      return Math.exp(-((Math.log(t) + 0.8) ** 2) / (2 * 0.35 ** 2)) / t;
    });
    expect(fitSigmaLognormal(bell)).toBeGreaterThan(0.7);
    expect(fitSigmaLognormal(Array(60).fill(10))).toBe(0);
  });
});

describe('keystrokes and scrolls', () => {
  test('emits physical codes and modifiers for shifted keys', () => {
    const plan = synthesizeKeystrokes('A!?{}', { mistakeRate: 0 }, 9);
    expect(plan.events.map(({ key, code, shiftKey }) => ({ key, code, shiftKey }))).toEqual([
      { key: 'A', code: 'KeyA', shiftKey: true },
      { key: '!', code: 'Digit1', shiftKey: true },
      { key: '?', code: 'Slash', shiftKey: true },
      { key: '{', code: 'BracketLeft', shiftKey: true },
      { key: '}', code: 'BracketRight', shiftKey: true },
    ]);
  });

  test('represents mistake correction pauses', () => {
    const plan = synthesizeKeystrokes('abcdef', { mistakeRate: 0.15 }, 8);
    const backspace = plan.events.findIndex((event) => event.key === 'Backspace');
    expect(backspace).toBeGreaterThan(0);
    const event = plan.events[backspace]!;
    expect(event.code).toBe('Backspace');
    expect(event.pauseAfterMs).toBeGreaterThanOrEqual(100);
    expect(event.pauseAfterMs).toBeLessThanOrEqual(300);
  });

  test('produces human-rate key rollover instead of serializing most key holds', () => {
    let pairs = 0;
    let overlaps = 0;
    for (let seed = 0; seed < 200; seed++) {
      const plan = synthesizeKeystrokes('the quick brown fox jumps over the lazy dog', { mistakeRate: 0 }, seed);
      for (let index = 1; index < plan.events.length; index++) {
        const previous = plan.events[index - 1]!;
        const current = plan.events[index]!;
        pairs++;
        if (current.downMs < previous.downMs + previous.holdMs) overlaps++;
      }
    }
    const ratio = overlaps / pairs;
    expect(ratio).toBeGreaterThan(0.35);
    expect(ratio).toBeLessThan(0.5);
  });

  test('preserves requested scroll distance and bounds short peaks', () => {
    for (const target of [-2000, -500, -179, -100, -25, -1, 0, 1, 25, 100, 179, 500, 2000]) {
      const plan = synthesizeScroll(target, undefined, 12);
      const sum = plan.frames.reduce((total, frame) => total + frame.deltaY, 0);
      expect(approx(sum, target, 1e-8)).toBe(true);
      if (Math.abs(target) < 180) expect(plan.peakDeltaY).toBeLessThanOrEqual(Math.abs(target));
    }
  });

  test('preserves 10,000px distances with repeated peak-bounded lobes', () => {
    for (const target of [-10_000, 10_000]) {
      const plan = synthesizeScroll(target, undefined, 12);
      const total = plan.frames.reduce((sum, frame) => sum + frame.deltaY, 0);
      const configuredPeak = Math.abs(plan.frames[0]!.deltaY);
      const peakFrames = plan.frames.filter((frame) => Math.abs(frame.deltaY) === configuredPeak);

      expect(approx(total, target, 1e-8)).toBe(true);
      expect(plan.frames.every((frame) => Math.abs(frame.deltaY) <= configuredPeak)).toBe(true);
      expect(plan.peakDeltaY).toBe(configuredPeak);
      expect(peakFrames.length).toBeGreaterThan(1);
    }
  });
});

describe('capture evidence extraction', () => {
  test('recovers timing and digraph variance without key content', () => {
    const capture: RawKeystroke = {
      sessionId: 'session',
      capturedAt: 1,
      sourceSchema: 'soma.capture.v2',
      trustedEvents: true,
      events: [
        { downMs: 220, upMs: 280, shiftKey: false, trusted: true, hand: 'right', keyKind: 'character', digraphClass: 'cross-hand' },
        { downMs: 0, upMs: 100, shiftKey: false, trusted: true, hand: 'left', keyKind: 'character', digraphClass: 'none' },
        { downMs: 400, upMs: 500, shiftKey: false, trusted: true, hand: 'right', keyKind: 'character', digraphClass: 'after-space' },
        { downMs: 120, upMs: 190, shiftKey: false, trusted: true, hand: 'left', keyKind: 'character', digraphClass: 'same-hand' },
      ],
      charCount: 4,
      charsCommitted: 4,
      inputTypes: ['insertText'],
      fieldType: 'text',
      correctionCount: 1,
    };
    const features = extractCapturedKeystrokeFeatures(capture);
    expect(features.hold_times).toEqual([100, 70, 60, 100]);
    expect(features.flight_times).toEqual([120, 100, 180]);
    expect(features.release_press_times).toEqual([20, 30, 120]);
    expect(features.overlap_ratio).toBe(0);
    expect(features.digraph_class_variance).toBeGreaterThan(0);
    expect(features.correction_count).toBe(1);
    expect(JSON.stringify(capture)).not.toContain('"key"');
    expect(JSON.stringify(capture)).not.toContain('"code"');
  });

  test('excludes modifiers from printable digraph timing', () => {
    const withModifier: RawKeystroke = {
      sessionId: 'session',
      capturedAt: 1,
      sourceSchema: 'soma.capture.v2',
      trustedEvents: true,
      events: [
        { downMs: 0, upMs: 80, shiftKey: false, trusted: true, hand: 'left', keyKind: 'character', digraphClass: 'none' },
        { downMs: 100, upMs: 140, shiftKey: true, trusted: true, hand: 'left', keyKind: 'modifier', digraphClass: 'same-hand' },
        { downMs: 150, upMs: 230, shiftKey: true, trusted: true, hand: 'right', keyKind: 'character', digraphClass: 'cross-hand' },
      ],
      charCount: 2,
      charsCommitted: 2,
      inputTypes: ['insertText', 'insertText'],
      fieldType: 'text',
      correctionCount: 0,
    };
    const features = extractCapturedKeystrokeFeatures(withModifier);
    expect(features.hold_times).toEqual([80, 80]);
    expect(features.flight_times).toEqual([150]);
    expect(features.release_press_times).toEqual([70]);
    expect(features.min_flight_time).toBe(150);
  });

  test('measures negative release-to-press latency as rollover', () => {
    const rollover: RawKeystroke = {
      sessionId: 'session',
      capturedAt: 1,
      sourceSchema: 'soma.capture.v2',
      trustedEvents: true,
      events: [
        { downMs: 0, upMs: 90, trusted: true, keyKind: 'character', digraphClass: 'none' },
        { downMs: 55, upMs: 130, trusted: true, keyKind: 'character', digraphClass: 'cross-hand' },
      ],
      charCount: 2,
      charsCommitted: 2,
      fieldType: 'text',
      correctionCount: 0,
    };
    const features = extractCapturedKeystrokeFeatures(rollover);
    expect(features.flight_times).toEqual([55]);
    expect(features.release_press_times).toEqual([-35]);
    expect(features.overlap_ratio).toBe(1);
  });

  test('derives provenance, cadence, and true action-ready latency', () => {
    const base = {
      sessionId: 'session',
      capturedAt: 1,
      sourceSchema: 'soma.capture.v2' as const,
      pointerType: 'mouse' as const,
    };
    const interactions: RawInteraction[] = [
      { ...base, eventType: 'keydown', capturedAt: 3, isTrusted: true, userActivation: true, pointerType: 'none', sinceNavigationMs: 500, actionReadySinceNavigationMs: 100, actionLatencyMs: 400 },
      { ...base, eventType: 'click', capturedAt: 2, isTrusted: true, userActivation: true, sinceNavigationMs: 300, actionReadySinceNavigationMs: 250, actionLatencyMs: 50 },
      { ...base, eventType: 'click', capturedAt: 4, isTrusted: false, userActivation: false, sinceNavigationMs: 550 },
      { ...base, eventType: 'wheel', capturedAt: 5, isTrusted: true, userActivation: true, sinceNavigationMs: 900 },
    ];
    const features = extractCapturedInteractionFeatures(interactions);
    expect(features.event.sequence).toEqual(['click', 'keydown', 'click', 'wheel']);
    expect(features.event.isTrusted).toBe(false);
    expect(features.event.has_user_activation).toBe(true);
    expect(features.interaction.count).toBe(1);  // one interaction (click@300) before last action boundary (keydown@500)
    expect(features.interaction.min_action_latency_ms).toBe(50);
    expect(features.interaction.action_rate).toBeGreaterThan(0);
    expect(extractCapturedInteractionFeatures([interactions[0]!]).interaction.count).toBe(0);  // single action: no prior interactions
    expect(extractCapturedInteractionFeatures(interactions, 500).interaction.count).toBe(1);  // explicit boundary: click@300 only
  });
});

describe('dispatch', () => {
  test('honors absolute trajectory, key, and scroll timelines', async () => {
    let elapsed = 0;
    const pointerEvents: Array<{ at: number; x: number; y: number }> = [];
    const trajectory = synthesizeMovement({ x: 0, y: 0 }, { x: 200, y: 100, width: 60, height: 30 }, undefined, 3);
    await dispatchTrajectory(
      trajectory,
      { move: async (x, y) => { pointerEvents.push({ at: elapsed, x, y }); } },
      async (milliseconds) => { elapsed += milliseconds; },
      () => elapsed,
    );
    expect(pointerEvents[0]!.at).toBe(trajectory.preMoveDelayMs);
    expect(pointerEvents.at(-1)!.at).toBeCloseTo(trajectory.preMoveDelayMs + trajectory.durationMs, 8);

    elapsed = 0;
    const keyEvents: Array<{ at: number; kind: string; key: string }> = [];
    const keys = synthesizeKeystrokes('ab', { mistakeRate: 0 }, 4);
    await dispatchKeystrokePlan(
      keys,
      {
        down: async (event) => { keyEvents.push({ at: elapsed, kind: 'down', key: event.key }); },
        up: async (event) => { keyEvents.push({ at: elapsed, kind: 'up', key: event.key }); },
      },
      async (milliseconds) => { elapsed += milliseconds; },
      () => elapsed,
    );
    expect(keyEvents.find((event) => event.kind === 'down' && event.key === 'b')!.at).toBeCloseTo(keys.events[1]!.downMs, 8);
    expect(keyEvents.at(-1)!.at).toBeCloseTo(keys.totalDurationMs, 8);

    elapsed = 0;
    const wheelEvents: Array<{ at: number; deltaY: number }> = [];
    const scroll = synthesizeScroll(500, undefined, 4);
    await dispatchScrollPlan(
      scroll,
      { wheel: async (deltaY) => { wheelEvents.push({ at: elapsed, deltaY }); } },
      async (milliseconds) => { elapsed += milliseconds; },
      () => elapsed,
    );
    expect(wheelEvents.at(-1)!.at).toBeCloseTo(scroll.durationMs, 8);
  });

  test('checks interactivity after movement and before pointer down', async () => {
    const order: string[] = [];
    let interactive = false;
    const pointer = new HumanPointer(
      {
        move: async () => { order.push('move'); interactive = true; },
        isInteractiveAt: async () => { order.push('hit-test'); return interactive; },
        down: async () => { order.push('down'); },
        up: async () => { order.push('up'); },
      },
      { x: 0, y: 0 },
      undefined,
      async () => undefined,
    );
    const result = await pointer.click({ x: 100, y: 100, width: 40, height: 20 }, 2);
    expect(result.clickTargetIsInteractive).toBe(true);
    expect(order.lastIndexOf('move')).toBeLessThan(order.indexOf('hit-test'));
    expect(order.indexOf('hit-test')).toBeLessThan(order.indexOf('down'));
  });

  test('refuses clicks when the live endpoint is not interactive', async () => {
    const pointer = new HumanPointer(
      {
        move: async () => undefined,
        isInteractiveAt: async () => false,
        down: async () => { throw new Error('must not dispatch'); },
        up: async () => undefined,
      },
      { x: 0, y: 0 },
      undefined,
      async () => undefined,
    );
    expect(pointer.click({ x: 100, y: 100, width: 40, height: 20 }, 2)).rejects.toThrow('not interactive');
  });
});

describe('scoring safety', () => {
  test('extracts finite canonical features', () => {
    const plan = synthesizeMovement({ x: 0, y: 0 }, { x: 300, y: 200, width: 80, height: 40 }, undefined, 4);
    const features = extractFeatureVector(plan.points, 340, 220, 40);
    expect(features).toHaveLength(12);
    expect(features.every((value) => Number.isFinite(value) && value >= 0 && value <= 1)).toBe(true);
  });

  test('does not infer tremor from a linear ramp', () => {
    const path = Array.from({ length: 600 }, (_, index) => ({
      x: index * 0.5,
      y: 0,
      tMs: index * 1000 / 60,
    }));
    expect(measureTremor(path).has_micro_tremor).toBe(false);
  });

  test('rejects malformed JSON weights without poisoning inference', () => {
    expect(validateModelWeights({})).toBe(false);
    expect(loadJSONModelText('{}')).toBe(false);
    expect(runJSONInference([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])).toBeNull();
  });

  test('seeded RNG is reproducible', () => {
    const a = new RNG(42);
    const b = new RNG(42);
    expect(Array.from({ length: 10 }, () => a.next())).toEqual(Array.from({ length: 10 }, () => b.next()));
  });
});

describe('extension routing', () => {
  test('injects capture into related opaque child frames', async () => {
    const manifest = await Bun.file('ext/manifest.json').json() as {
      content_scripts: Array<{ all_frames?: boolean; match_about_blank?: boolean; match_origin_as_fallback?: boolean }>;
    };
    expect(manifest.content_scripts[0]).toMatchObject({
      all_frames: true,
      match_about_blank: true,
      match_origin_as_fallback: true,
    });
  });
});

describe('flow synthesis', () => {
  test('caps dispatch frames without truncating long movement duration', async () => {
    const { MAX_DISPATCH_FRAMES, resampleToFrameRate } = await import('../src/pointer/flow-synthesis.js');
    const source = {
      u: new Float64Array([0, 1]),
      v: new Float64Array([0, 0]),
      t: new Float64Array([0, 3_600_000]),
    };
    const output = resampleToFrameRate(source.u, source.v, source.t, 1000 / 60);
    expect(output.t.length).toBe(MAX_DISPATCH_FRAMES);
    expect(output.t[output.t.length - 1]).toBe(3_600_000);
    expect(output.u[output.u.length - 1]).toBe(1);
  });

  test('reports the persona-scaled flow duration', async () => {
    const { synthesizeMovementFlow } = await import('../src/pointer/flow-synthesis.js');
    const model = {
      stats: {
        schemaVersion: 2 as const,
        arch: { dataDim: 3, contextDim: 3, nTransforms: 1, nBins: 2, tailBound: 6 },
        layout: {
          nSteps: 2,
          nFree: 1,
          droppedDeltaIndex: 1,
          duStart: 0,
          dvStart: 1,
          logDurationIndex: 2,
        },
        stats: {
          xMean: [0.5, 0, Math.log(100)],
          xStd: [1, 1, 1],
          cMean: [0, 0, 0],
          cStd: [1, 1, 1],
        },
        postprocess: { terminalStopProbability: 1, terminalStopEpsilon: 1e-6 },
      },
      session: {
        run: async () => ({ x: { data: new Float32Array([0, 0, 0]) } }),
      },
      tensor: (data: Float32Array) => ({ data }),
    };
    const plan = await synthesizeMovementFlow(
      model,
      { x: 0, y: 0 },
      { x: 100, y: 0, width: 20, height: 20 },
      { speed: 2, tremor: 0, precision: 1 },
      42,
    );
    expect(Math.abs(plan.durationMs - 50)).toBeLessThan(1e-4);
    expect(plan.points[plan.points.length - 1]!.tMs).toBe(plan.durationMs);
    expect(plan.points[plan.points.length - 2]!.x).toBe(plan.points[plan.points.length - 1]!.x);
    expect(plan.points[plan.points.length - 2]!.y).toBe(plan.points[plan.points.length - 1]!.y);
  });

  test('reconstructGesture is deterministic and endpoint-exact', async () => {
    // Test without ONNX — exercises the reconstruction math only.
    const { reconstructGesture, validateFlowStats } = await import('../src/index.js');
    const statsRaw = await Bun.file(new URL('../model/flow.stats.json', import.meta.url)).text()
      .catch(() => null);
    if (!statsRaw) return; // skip if model not built yet

    const stats = JSON.parse(statsRaw);
    validateFlowStats(stats);

    const dim = stats.arch.dataDim;
    // Determinism: same x → same output
    const x1 = new Float32Array(dim).fill(0);
    const x2 = new Float32Array(dim).fill(0);
    const r1 = reconstructGesture(x1, stats);
    const r2 = reconstructGesture(x2, stats);
    expect(Array.from(r1.u)).toEqual(Array.from(r2.u));
    expect(Array.from(r1.v)).toEqual(Array.from(r2.v));
    expect(Array.from(r1.t)).toEqual(Array.from(r2.t));

    // Endpoint constraint: sum(du) == 1, sum(dv) == 0 in the canonical frame
    const n = r1.u.length;
    expect(r1.u[0]).toBe(0);
    expect(r1.v[0]).toBe(0);
    expect(Math.abs(r1.u[n - 1]! - 1.0)).toBeLessThan(1e-6);
    expect(Math.abs(r1.v[n - 1]!)).toBeLessThan(1e-6);

    const stopped = reconstructGesture(x1, stats, true);
    expect(stopped.u[stopped.u.length - 1]! - stopped.u[stopped.u.length - 2]!).toBe(0);
    expect(stopped.v[stopped.v.length - 1]! - stopped.v[stopped.v.length - 2]!).toBe(0);

    // Time is monotonically non-decreasing
    for (let i = 1; i < r1.t.length; i++) {
      expect(r1.t[i]!).toBeGreaterThanOrEqual(r1.t[i - 1]!);
    }

    // durationMs is positive and finite
    expect(r1.durationMs).toBeGreaterThan(0);
    expect(Number.isFinite(r1.durationMs)).toBe(true);
  });
});
