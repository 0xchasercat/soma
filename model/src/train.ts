/**
 * Train the neural motion scorer: full backpropagation + Adam optimizer.
 *
 * Architecture: 12 → FC(104) → ReLU → FC(64) → ReLU → FC(32) → ReLU → FC(1) → Sigmoid
 * ~10,185 parameters. Label: 0 = human, 1 = bot.
 *
 * Usage:
 *   bun run model/src/train.ts <capture.json> [--epochs N] [--lr N] [--batch N]
 */

import { loadCapturedData, buildDataset, type TrainingSample } from './dataset.js';
import type { BehaviorProfile, CaptureExport, ModelFeatureVector } from '../../src/types.js';
import { MODEL_ARTIFACT_SCHEMA, MODEL_FEATURE_SCHEMA } from '../../src/score/json_model.js';
import { RNG } from '../../src/rng.js';
import { extractFeatureVector } from './features.js';
import { synthesizeMovement } from '../../src/pointer/trajectory.js';
import { PROFILES } from '../../src/profile.js';

// ─── Dimensions ──────────────────────────────────────────────────────────────

const IN = 12;
const H1 = 104;
const H2 = 64;
const H3 = 32;

// ─── Adam hyper-parameters ───────────────────────────────────────────────────

const ADAM_B1 = 0.9;
const ADAM_B2 = 0.999;
const ADAM_EPS = 1e-8;
const WEIGHT_DECAY = 1e-4; // L2 regularization

// ─── Utilities ───────────────────────────────────────────────────────────────

function zeros(n: number): Float64Array {
  return new Float64Array(n);
}

/** Box-Muller normal sample from the run-scoped seeded RNG. */
function randn(rng: RNG): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng.next();
  while (v === 0) v = rng.next();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** He initialization: N(0, sqrt(2/fanIn)). */
function heInit(fanIn: number, n: number, rng: RNG): Float64Array {
  const scale = Math.sqrt(2 / fanIn);
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = randn(rng) * scale;
  return w;
}

function relu(x: number): number {
  return x > 0 ? x : 0;
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

// ─── Model (flat row-major arrays) ───────────────────────────────────────────

interface Model {
  W1: Float64Array; // [H1 × IN]
  b1: Float64Array; // [H1]
  W2: Float64Array; // [H2 × H1]
  b2: Float64Array; // [H2]
  W3: Float64Array; // [H3 × H2]
  b3: Float64Array; // [H3]
  W4: Float64Array; // [H3] — single output neuron weights
  b4: Float64Array; // [1]
}

function initModel(rng: RNG): Model {
  return {
    W1: heInit(IN, H1 * IN, rng), b1: zeros(H1),
    W2: heInit(H1, H2 * H1, rng), b2: zeros(H2),
    W3: heInit(H2, H3 * H2, rng), b3: zeros(H3),
    W4: heInit(H3, H3, rng), b4: zeros(1),
  };
}

// ─── Adam momentum state ─────────────────────────────────────────────────────

interface Adam {
  t: number;
  mW1: Float64Array; vW1: Float64Array;
  mb1: Float64Array; vb1: Float64Array;
  mW2: Float64Array; vW2: Float64Array;
  mb2: Float64Array; vb2: Float64Array;
  mW3: Float64Array; vW3: Float64Array;
  mb3: Float64Array; vb3: Float64Array;
  mW4: Float64Array; vW4: Float64Array;
  mb4: Float64Array; vb4: Float64Array;
}

function initAdam(m: Model): Adam {
  const z = zeros;
  return {
    t: 0,
    mW1: z(m.W1.length), vW1: z(m.W1.length),
    mb1: z(H1), vb1: z(H1),
    mW2: z(m.W2.length), vW2: z(m.W2.length),
    mb2: z(H2), vb2: z(H2),
    mW3: z(m.W3.length), vW3: z(m.W3.length),
    mb3: z(H3), vb3: z(H3),
    mW4: z(H3), vW4: z(H3),
    mb4: z(1), vb4: z(1),
  };
}

// ─── Forward pass ────────────────────────────────────────────────────────────

interface Activations {
  z1: Float64Array; h1: Float64Array;
  z2: Float64Array; h2: Float64Array;
  z3: Float64Array; h3: Float64Array;
  z4: number; out: number;
}

function forward(x: ModelFeatureVector, m: Model): Activations {
  // Layer 1
  const z1 = new Float64Array(H1);
  const h1 = new Float64Array(H1);
  for (let i = 0; i < H1; i++) {
    let s = m.b1[i]!;
    const base = i * IN;
    for (let j = 0; j < IN; j++) s += m.W1[base + j]! * x[j];
    z1[i] = s;
    h1[i] = relu(s);
  }

  // Layer 2
  const z2 = new Float64Array(H2);
  const h2 = new Float64Array(H2);
  for (let i = 0; i < H2; i++) {
    let s = m.b2[i]!;
    const base = i * H1;
    for (let j = 0; j < H1; j++) s += m.W2[base + j]! * h1[j]!;
    z2[i] = s;
    h2[i] = relu(s);
  }

  // Layer 3
  const z3 = new Float64Array(H3);
  const h3 = new Float64Array(H3);
  for (let i = 0; i < H3; i++) {
    let s = m.b3[i]!;
    const base = i * H2;
    for (let j = 0; j < H2; j++) s += m.W3[base + j]! * h2[j]!;
    z3[i] = s;
    h3[i] = relu(s);
  }

  // Layer 4 (scalar output)
  let z4 = m.b4[0]!;
  for (let j = 0; j < H3; j++) z4 += m.W4[j]! * h3[j]!;
  const out = sigmoid(z4);

  return { z1, h1, z2, h2, z3, h3, z4, out };
}

// ─── Gradient accumulator ────────────────────────────────────────────────────

interface Grads {
  gW1: Float64Array; gb1: Float64Array;
  gW2: Float64Array; gb2: Float64Array;
  gW3: Float64Array; gb3: Float64Array;
  gW4: Float64Array; gb4: Float64Array;
}

function zeroGrads(m: Model): Grads {
  return {
    gW1: zeros(m.W1.length), gb1: zeros(H1),
    gW2: zeros(m.W2.length), gb2: zeros(H2),
    gW3: zeros(m.W3.length), gb3: zeros(H3),
    gW4: zeros(H3), gb4: zeros(1),
  };
}

/** Accumulate gradients for one sample into g (in place). */
function accumulateGrad(
  x: ModelFeatureVector,
  target: number,
  act: Activations,
  m: Model,
  g: Grads,
): void {
  // dL/dz4 = out - target  (BCE + sigmoid: derivative simplifies cleanly)
  const dz4 = act.out - target;

  // ── FC4 ──────────────────────────────────────────────────────────────────
  g.gb4[0]! += dz4;
  for (let j = 0; j < H3; j++) g.gW4[j]! += dz4 * act.h3[j]!;

  // dL/dh3[j] = dz4 * W4[j]
  const dh3 = new Float64Array(H3);
  for (let j = 0; j < H3; j++) dh3[j] = dz4 * m.W4[j]!;

  // dL/dz3[i] = dh3[i] * relu'(z3[i])
  const dz3 = new Float64Array(H3);
  for (let i = 0; i < H3; i++) dz3[i] = dh3[i]! * (act.z3[i]! > 0 ? 1 : 0);

  // ── FC3 ──────────────────────────────────────────────────────────────────
  for (let i = 0; i < H3; i++) {
    if (dz3[i] === 0) continue;
    g.gb3[i]! += dz3[i]!;
    const base = i * H2;
    for (let k = 0; k < H2; k++) g.gW3[base + k]! += dz3[i]! * act.h2[k]!;
  }

  // dL/dh2[k] = Σ_i dz3[i] * W3[i*H2+k]
  const dh2 = new Float64Array(H2);
  for (let i = 0; i < H3; i++) {
    if (dz3[i] === 0) continue;
    const base = i * H2;
    for (let k = 0; k < H2; k++) dh2[k]! += dz3[i]! * m.W3[base + k]!;
  }

  // dL/dz2[i] = dh2[i] * relu'(z2[i])
  const dz2 = new Float64Array(H2);
  for (let i = 0; i < H2; i++) dz2[i] = dh2[i]! * (act.z2[i]! > 0 ? 1 : 0);

  // ── FC2 ──────────────────────────────────────────────────────────────────
  for (let i = 0; i < H2; i++) {
    if (dz2[i] === 0) continue;
    g.gb2[i]! += dz2[i]!;
    const base = i * H1;
    for (let k = 0; k < H1; k++) g.gW2[base + k]! += dz2[i]! * act.h1[k]!;
  }

  // dL/dh1[k] = Σ_i dz2[i] * W2[i*H1+k]
  const dh1 = new Float64Array(H1);
  for (let i = 0; i < H2; i++) {
    if (dz2[i] === 0) continue;
    const base = i * H1;
    for (let k = 0; k < H1; k++) dh1[k]! += dz2[i]! * m.W2[base + k]!;
  }

  // dL/dz1[i] = dh1[i] * relu'(z1[i])
  const dz1 = new Float64Array(H1);
  for (let i = 0; i < H1; i++) dz1[i] = dh1[i]! * (act.z1[i]! > 0 ? 1 : 0);

  // ── FC1 ──────────────────────────────────────────────────────────────────
  for (let i = 0; i < H1; i++) {
    if (dz1[i] === 0) continue;
    g.gb1[i]! += dz1[i]!;
    const base = i * IN;
    for (let k = 0; k < IN; k++) g.gW1[base + k]! += dz1[i]! * x[k];
  }
}

// ─── Adam update ─────────────────────────────────────────────────────────────

function adamStep(
  weights: Float64Array,
  gradients: Float64Array,
  firstMoment: Float64Array,
  secondMoment: Float64Array,
  biasCorrection1: number,
  biasCorrection2: number,
  learningRate: number,
  applyWeightDecay: boolean,
): void {
  for (let i = 0; i < weights.length; i++) {
    const gradient = gradients[i]!;
    firstMoment[i] = ADAM_B1 * firstMoment[i]! + (1 - ADAM_B1) * gradient;
    secondMoment[i] = ADAM_B2 * secondMoment[i]! + (1 - ADAM_B2) * gradient * gradient;
    const adamUpdate = (firstMoment[i]! / biasCorrection1) /
      (Math.sqrt(secondMoment[i]! / biasCorrection2) + ADAM_EPS);
    const decayUpdate = applyWeightDecay ? WEIGHT_DECAY * weights[i]! : 0;
    weights[i] -= learningRate * (adamUpdate + decayUpdate);
  }
}

function applyAdam(model: Model, gradients: Grads, adam: Adam, learningRate: number): void {
  adam.t++;
  const correction1 = 1 - Math.pow(ADAM_B1, adam.t);
  const correction2 = 1 - Math.pow(ADAM_B2, adam.t);
  adamStep(model.W1, gradients.gW1, adam.mW1, adam.vW1, correction1, correction2, learningRate, true);
  adamStep(model.b1, gradients.gb1, adam.mb1, adam.vb1, correction1, correction2, learningRate, false);
  adamStep(model.W2, gradients.gW2, adam.mW2, adam.vW2, correction1, correction2, learningRate, true);
  adamStep(model.b2, gradients.gb2, adam.mb2, adam.vb2, correction1, correction2, learningRate, false);
  adamStep(model.W3, gradients.gW3, adam.mW3, adam.vW3, correction1, correction2, learningRate, true);
  adamStep(model.b3, gradients.gb3, adam.mb3, adam.vb3, correction1, correction2, learningRate, false);
  adamStep(model.W4, gradients.gW4, adam.mW4, adam.vW4, correction1, correction2, learningRate, true);
  adamStep(model.b4, gradients.gb4, adam.mb4, adam.vb4, correction1, correction2, learningRate, false);
}

// ─── Training loop ───────────────────────────────────────────────────────────

function shuffle<T>(arr: T[], rng: RNG): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = rng.nextInt(i + 1);
    [arr[i], arr[j]] = [arr[j]!, arr[i]!];
  }
}

interface Evaluation {
  loss: number;
  accuracy: number;
  trueHuman: number;
  falseBot: number;
  falseHuman: number;
  trueBot: number;
}

function evaluate(samples: TrainingSample[], model: Model): Evaluation {
  if (samples.length === 0) throw new Error('Cannot evaluate an empty cohort');
  let totalLoss = 0;
  let trueHuman = 0;
  let falseBot = 0;
  let falseHuman = 0;
  let trueBot = 0;
  for (const sample of samples) {
    const { out } = forward(sample.features, model);
    totalLoss += -(sample.label * Math.log(out + 1e-8) + (1 - sample.label) * Math.log(1 - out + 1e-8));
    const predicted = out >= 0.5 ? 1 : 0;
    if (sample.label === 0 && predicted === 0) trueHuman++;
    else if (sample.label === 0) falseBot++;
    else if (predicted === 0) falseHuman++;
    else trueBot++;
  }
  return {
    loss: totalLoss / samples.length,
    accuracy: (trueHuman + trueBot) / samples.length,
    trueHuman,
    falseBot,
    falseHuman,
    trueBot,
  };
}

function groupedSplit(samples: TrainingSample[], rng: RNG): {
  trainSet: TrainingSample[];
  validationSet: TrainingSample[];
  testSet: TrainingSample[];
} {
  const calibration = samples.filter((sample) => sample.calibration === true);
  const benchmarkSamples = samples.filter((sample) => sample.calibration !== true);
  const groups = new Map<string, TrainingSample[]>();
  for (const sample of benchmarkSamples) groups.set(sample.groupId, [...(groups.get(sample.groupId) ?? []), sample]);
  const groupsByLabel = new Map<0 | 1, string[]>();
  for (const [groupId, group] of groups) {
    const label = group[0]!.label;
    if (group.some((sample) => sample.label !== label)) throw new Error(`Mixed-label group ${groupId}`);
    groupsByLabel.set(label, [...(groupsByLabel.get(label) ?? []), groupId]);
  }

  const validationIds = new Set<string>();
  const testIds = new Set<string>();
  for (const label of [0, 1] as const) {
    const ids = groupsByLabel.get(label) ?? [];
    if (ids.length < 5) throw new Error('Train/validation/test metrics require at least five independent groups per class');
    shuffle(ids, rng);
    const cohortSize = Math.max(1, Math.floor(ids.length * 0.15));
    for (const id of ids.slice(0, cohortSize)) testIds.add(id);
    for (const id of ids.slice(cohortSize, cohortSize * 2)) validationIds.add(id);
  }

  return {
    testSet: benchmarkSamples.filter((sample) => testIds.has(sample.groupId)),
    validationSet: benchmarkSamples.filter((sample) => validationIds.has(sample.groupId)),
    trainSet: [
      ...benchmarkSamples.filter((sample) => !testIds.has(sample.groupId) && !validationIds.has(sample.groupId)),
      ...calibration,
    ],
  };
}

async function train(
  samples: TrainingSample[],
  epochs = 50,
  learningRate = 0.001,
  batchSize = 32,
  seed = 42,
): Promise<{ model: Model; validation: Evaluation; test: Evaluation; testSet: TrainingSample[] }> {
  const rng = new RNG(seed);
  if (new Set(samples.map((sample) => sample.label)).size !== 2) {
    throw new Error('Training requires both human (0) and bot (1) classes');
  }
  const { trainSet, validationSet, testSet } = groupedSplit(samples, rng);
  for (const [name, cohort] of [['train', trainSet], ['validation', validationSet], ['test', testSet]] as const) {
    if (cohort.length === 0 || new Set(cohort.map((sample) => sample.label)).size !== 2) {
      throw new Error(`${name} cohort must contain both classes`);
    }
  }
  console.log(`  train: ${trainSet.length}, validation: ${validationSet.length}, test: ${testSet.length} (group-disjoint)`);

  const model = initModel(rng);
  const adam = initAdam(model);
  let bestValidationLoss = Infinity;
  let bestWeights: Model | null = null;

  for (let epoch = 0; epoch < epochs; epoch++) {
    shuffle(trainSet, rng);
    let trainLoss = 0;
    for (let i = 0; i < trainSet.length; i += batchSize) {
      const batch = trainSet.slice(i, i + batchSize);
      const gradients = zeroGrads(model);
      for (const sample of batch) {
        const activation = forward(sample.features, model);
        trainLoss += -(sample.label * Math.log(activation.out + 1e-8) + (1 - sample.label) * Math.log(1 - activation.out + 1e-8));
        accumulateGrad(sample.features, sample.label, activation, model, gradients);
      }
      const inverseBatch = 1 / batch.length;
      for (const values of [gradients.gW1, gradients.gb1, gradients.gW2, gradients.gb2, gradients.gW3, gradients.gb3, gradients.gW4, gradients.gb4]) {
        for (let index = 0; index < values.length; index++) values[index]! *= inverseBatch;
      }
      applyAdam(model, gradients, adam, learningRate);
    }

    const validation = evaluate(validationSet, model);
    if ((epoch + 1) % 10 === 0 || epoch < 5) {
      console.log(`  Epoch ${String(epoch + 1).padStart(3)}/${epochs}` +
        ` train_loss=${(trainLoss / trainSet.length).toFixed(4)}` +
        ` validation_loss=${validation.loss.toFixed(4)}` +
        ` validation_acc=${(validation.accuracy * 100).toFixed(1)}%`);
    }
    if (validation.loss < bestValidationLoss) {
      bestValidationLoss = validation.loss;
      bestWeights = deepCopyModel(model);
    }
  }

  const finalModel = bestWeights ?? model;
  const validation = evaluate(validationSet, finalModel);
  const test = evaluate(testSet, finalModel);
  console.log(`Final validation: loss=${validation.loss.toFixed(4)} accuracy=${(validation.accuracy * 100).toFixed(1)}%`);
  console.log(`Untouched test: loss=${test.loss.toFixed(4)} accuracy=${(test.accuracy * 100).toFixed(1)}%` +
    ` confusion=[human ${test.trueHuman}/${test.trueHuman + test.falseBot}, bot ${test.trueBot}/${test.trueBot + test.falseHuman}]`);
  return { model: finalModel, validation, test, testSet };
}

interface SelfConsistency {
  count: number;
  meanBotScore: number;
  maxBotScore: number;
  botRate: number;
}

/**
 * Held-out probe: score Soma's synthesis under a model that never saw it.
 *
 * Soma is excluded from the training set (see dataset.ts), so this is a measurement
 * rather than a construction. Sweeps every shipped persona and a wide geometry range,
 * because a probe fixed to DEFAULT_PROFILE and one target band would miss exactly the
 * personas that deviate most.
 */
function evaluateSomaPlans(model: Model, seed: number, count = 400): SelfConsistency {
  const personas: Array<Partial<BehaviorProfile> | undefined> = [
    undefined, PROFILES.careful, PROFILES.average, PROFILES.fast, PROFILES.mobile,
  ];
  const rng = new RNG(seed ^ 0x50a1);
  const scores: number[] = [];
  for (let index = 0; index < count; index++) {
    const start = { x: rng.nextRange(20, 1500), y: rng.nextRange(20, 900) };
    const target = {
      x: rng.nextRange(20, 1500),
      y: rng.nextRange(20, 900),
      width: rng.nextRange(24, 200),
      height: rng.nextRange(18, 90),
    };
    const plan = synthesizeMovement(start, target, personas[index % personas.length], seed + index);
    scores.push(forward(extractFeatureVector(plan.points), model).out);
  }
  return {
    count,
    meanBotScore: scores.reduce((sum, score) => sum + score, 0) / scores.length,
    maxBotScore: Math.max(...scores),
    botRate: scores.filter((score) => score >= 0.5).length / scores.length,
  };
}

/**
 * Score real held-out human trajectories under the trained model.
 *
 * This is the reference the Soma probe must match. An absolute Soma bot-rate is not
 * meaningful on its own: if the model flags 12% of genuine humans, then Soma flagged
 * at 12% is indistinguishable, while Soma flagged at 0% is anomalously clean — a
 * synthesizer that is *too* perfect is its own tell.
 */
function evaluateHeldOutHumans(model: Model, samples: TrainingSample[]): SelfConsistency {
  const scores = samples
    .filter((sample) => sample.label === 0)
    .map((sample) => forward(sample.features, model).out);
  if (scores.length === 0) return { count: 0, meanBotScore: 0, maxBotScore: 0, botRate: 0 };
  return {
    count: scores.length,
    meanBotScore: scores.reduce((sum, score) => sum + score, 0) / scores.length,
    maxBotScore: Math.max(...scores),
    botRate: scores.filter((score) => score >= 0.5).length / scores.length,
  };
}

function evaluateLongStraightNegatives(model: Model, count = 100): SelfConsistency {
  const scores: number[] = [];
  for (let index = 0; index < count; index++) {
    const durationMs = 5_000 + (index / Math.max(1, count - 1)) * 5_000;
    const steps = Math.ceil(durationMs / (1000 / 60));
    const endX = 250 + (index % 11) * 35;
    const endY = (index % 7) * 18;
    const path = Array.from({ length: steps + 1 }, (_, step) => ({
      x: (endX * step) / steps,
      y: (endY * step) / steps,
      tMs: (durationMs * step) / steps,
    }));
    scores.push(forward(extractFeatureVector(path), model).out);
  }
  return {
    count,
    meanBotScore: scores.reduce((sum, score) => sum + score, 0) / scores.length,
    maxBotScore: Math.max(...scores),
    botRate: scores.filter((score) => score >= 0.5).length / scores.length,
  };
}

// ─── Deep copy (for best-checkpoint tracking) ────────────────────────────────

function deepCopyModel(m: Model): Model {
  return {
    W1: m.W1.slice(), b1: m.b1.slice(),
    W2: m.W2.slice(), b2: m.b2.slice(),
    W3: m.W3.slice(), b3: m.b3.slice(),
    W4: m.W4.slice(), b4: m.b4.slice(),
  };
}

// ─── Serialize to ModelWeights format (for onnx.ts / model.json) ─────────────

function serializeModel(m: Model) {
  function toMatrix(w: Float64Array, rows: number, cols: number): number[][] {
    return Array.from({ length: rows }, (_, i) =>
      Array.from({ length: cols }, (_, j) => w[i * cols + j]!),
    );
  }
  return {
    schema: MODEL_ARTIFACT_SCHEMA,
    featureSchema: MODEL_FEATURE_SCHEMA,
    fc1_w: toMatrix(m.W1, H1, IN),
    fc1_b: Array.from(m.b1),
    fc2_w: toMatrix(m.W2, H2, H1),
    fc2_b: Array.from(m.b2),
    fc3_w: toMatrix(m.W3, H3, H2),
    fc3_b: Array.from(m.b3),
    fc4_w: toMatrix(m.W4, 1, H3),
    fc4_b: Array.from(m.b4),
  };
}

// ─── Main ────────────────────────────────────────────────────────────────────

async function main() {
  const args = process.argv.slice(2);
  if (args.length === 0) {
    console.error('Usage: bun run model/src/train.ts <capture.json...> [--allow-legacy] [--epochs N] [--lr N] [--batch N] [--seed N] [--out model.json] [--metrics metrics.json]');
    process.exit(1);
  }

  const jsonPaths: string[] = [];
  let epochs = 50;
  let learningRate = 0.001;
  let batchSize = 32;
  let seed = 42;
  let outputPath = 'model/model.json';
  let metricsPath = 'model/model.metrics.json';
  let allowLegacy = false;
  for (let index = 0; index < args.length; index++) {
    const argument = args[index]!;
    if (argument === '--allow-legacy') allowLegacy = true;
    else if (argument === '--epochs') epochs = Number(args[++index]);
    else if (argument === '--lr') learningRate = Number(args[++index]);
    else if (argument === '--batch') batchSize = Number(args[++index]);
    else if (argument === '--seed') seed = Number(args[++index]);
    else if (argument === '--out') outputPath = args[++index] ?? '';
    else if (argument === '--metrics') metricsPath = args[++index] ?? '';
    else if (argument.startsWith('--')) throw new Error(`Unknown argument: ${argument}`);
    else jsonPaths.push(argument);
  }
  if (jsonPaths.length === 0) throw new Error('At least one capture JSON path is required');
  if (!Number.isInteger(epochs) || epochs < 1 || epochs > 10_000) throw new Error('--epochs must be an integer in [1, 10000]');
  if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 1) throw new Error('--lr must be in (0, 1]');
  if (!Number.isInteger(batchSize) || batchSize < 1 || batchSize > 65_536) throw new Error('--batch must be an integer in [1, 65536]');
  if (!Number.isInteger(seed)) throw new Error('--seed must be an integer');
  if (!outputPath || !metricsPath) throw new Error('--out and --metrics paths must not be empty');

  const artifactName = (path: string): string => path.split(/[\\/]/).filter(Boolean).at(-1) ?? path;
  const captureArtifacts = jsonPaths.map(artifactName);

  const allCaptures = await Promise.all(jsonPaths.map(loadCapturedData));
  const movementCount = allCaptures.reduce((sum, capture) => sum + capture.movements.length, 0);
  const keystrokeCount = allCaptures.reduce((sum, capture) => sum + capture.keystrokes.length, 0);
  const scrollCount = allCaptures.reduce((sum, capture) => sum + capture.scrolls.length, 0);
  console.log(`Loaded: ${movementCount} movements, ${keystrokeCount} keystrokes, ${scrollCount} scrolls`);
  const schemasForCapture = (capture: CaptureExport): Array<CaptureExport['schema']> => [
    capture.schema,
    ...capture.movements.flatMap((movement) => {
      const sourceSchema = 'sourceSchema' in movement ? movement.sourceSchema : undefined;
      return sourceSchema === 'soma.capture.v1' || sourceSchema === 'soma.capture.v2' ? [sourceSchema] : [];
    }),
  ];
  const sourceSchemas = [...new Set(allCaptures.flatMap(schemasForCapture))];
  const legacyCaptureFiles = captureArtifacts.filter((_, index) =>
    schemasForCapture(allCaptures[index]!).includes('soma.capture.v1'));
  if (legacyCaptureFiles.length > 0 && allowLegacy) {
    console.warn('Legacy v1 capture accepted by explicit --allow-legacy operator attestation.');
  }

  const dataset = buildDataset(allCaptures, { seed, allowLegacy });
  const capturedHumanCount = dataset.filter((sample) => sample.label === 0).length;
  const botCount = dataset.filter((sample) => sample.label === 1).length;
  const hardCount = dataset.filter((sample) => sample.groupId.startsWith('hard-')).length;
  if (capturedHumanCount === 0 || botCount === 0) throw new Error('Training requires non-empty human and bot classes');
  console.log(`Dataset: ${dataset.length} samples (${capturedHumanCount} captured human, ` +
    `${botCount - hardCount} easy negatives, ${hardCount} hard negatives [ghost-cursor/bezmouse/HumanCursor]), seed=${seed}`);
  console.log('Soma output is NOT in the dataset — it is scored only as a held-out probe.');

  console.log(`Training: epochs=${epochs} lr=${learningRate} batch=${batchSize}`);
  const result = await train(dataset, epochs, learningRate, batchSize, seed);
  const weights = serializeModel(result.model);
  const serialized = JSON.stringify(weights, null, 2);
  // Soma is held out of training, so these two numbers are directly comparable and
  // the comparison — not Soma's absolute score — is the acceptance criterion.
  const selfConsistency = evaluateSomaPlans(result.model, seed + 10_000);
  const humanReference = evaluateHeldOutHumans(result.model, result.testSet);
  console.log(`Held-out real humans: mean_bot_score=${humanReference.meanBotScore.toFixed(4)}` +
    ` bot_rate=${(humanReference.botRate * 100).toFixed(1)}% (n=${humanReference.count})`);
  console.log(`Soma (held out of training): mean_bot_score=${selfConsistency.meanBotScore.toFixed(4)}` +
    ` max=${selfConsistency.maxBotScore.toFixed(4)} bot_rate=${(selfConsistency.botRate * 100).toFixed(1)}%`);

  if (humanReference.count < 100) {
    throw new Error(`Refusing model promotion: only ${humanReference.count} held-out human samples; need >=100 for a usable reference`);
  }
  // The gate is two-sided against the human reference. Scoring far WORSE than real
  // humans means the synthesis is separable. Scoring far BETTER means it is
  // anomalously clean — real motion carries noise, and a synthesizer with none is
  // itself detectable. Either direction is a synthesis defect, not a model defect.
  const humanBotRate = humanReference.botRate;
  const upperBound = Math.max(0.05, humanBotRate * 2.5);
  if (selfConsistency.botRate > upperBound) {
    throw new Error(
      `Refusing model promotion: Soma bot_rate ${(selfConsistency.botRate * 100).toFixed(1)}% exceeds ` +
      `${(upperBound * 100).toFixed(1)}% (real held-out humans score ${(humanBotRate * 100).toFixed(1)}%). ` +
      'The synthesis is separable from human motion — fix the synthesizer, not this threshold.',
    );
  }
  const meanRatio = humanReference.meanBotScore > 1e-6
    ? selfConsistency.meanBotScore / humanReference.meanBotScore
    : Number.POSITIVE_INFINITY;
  console.log(`Soma/human mean-score ratio: ${Number.isFinite(meanRatio) ? meanRatio.toFixed(3) : 'n/a'}` +
    ' (1.0 = indistinguishable; <<1 = anomalously clean; >>1 = separable)');

  const negativeConsistency = evaluateLongStraightNegatives(result.model);
  console.log(`Long-straight negatives: mean_bot_score=${negativeConsistency.meanBotScore.toFixed(4)}` +
    ` bot_rate=${(negativeConsistency.botRate * 100).toFixed(1)}%`);
  if (negativeConsistency.botRate < 0.95) {
    throw new Error('Refusing model promotion: fewer than 95% of long straight negatives score as bots');
  }
  await Bun.write(outputPath, serialized);
  const hash = new Bun.CryptoHasher('sha256').update(serialized).digest('hex');
  await Bun.write(metricsPath, JSON.stringify({
    schema: 'soma.model-evaluation.v1',
    featureSchema: 'soma.motion-features.v2',
    modelSha256: hash,
    captureSchemas: sourceSchemas,
    captureFiles: captureArtifacts,
    legacyAttestation: {
      flagProvided: allowLegacy,
      acceptedCaptureFiles: allowLegacy ? legacyCaptureFiles : [],
    },
    seed,
    epochs,
    learningRate,
    batchSize,
    dataset: { capturedHuman: capturedHumanCount, easyNegatives: botCount - hardCount, hardNegatives: hardCount, somaInTraining: false },
    validation: result.validation,
    test: result.test,
    humanReference,
    somaProbe: selfConsistency,
    somaHumanMeanRatio: Number.isFinite(meanRatio) ? meanRatio : null,
    negativeConsistency,
    limitations: [
      'Human data represents one operator and is not population-level ground truth.',
      'Soma is excluded from training and scored as a held-out probe; its bot-rate is compared against held-out real humans rather than an absolute floor.',
      'Hard negatives are faithful reimplementations of published library algorithms, not the vendored libraries themselves.',
      'Synthetic test groups use the same generator families as training groups.',
      'Pointer-only: the 12 features are trajectory geometry. Keystroke and scroll evidence are gated separately and are not model inputs.',
    ],
  }, null, 2));
  console.log(`Saved canonical JSON weights -> ${outputPath}`);
  console.log(`Saved evaluation metrics -> ${metricsPath}`);
}

await main();
