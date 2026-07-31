/** Explicit JSON model loader and inference. JSON is the canonical model format. */

import type { ModelFeatureVector } from '../types.js';

export interface ModelWeights {
  fc1_w: number[][]; fc1_b: number[];
  fc2_w: number[][]; fc2_b: number[];
  fc3_w: number[][]; fc3_b: number[];
  fc4_w: number[][]; fc4_b: number[];
}

export const MODEL_ARTIFACT_SCHEMA = 'soma.motion-model.v1';
export const MODEL_FEATURE_SCHEMA = 'soma.motion-features.v2';
export interface CanonicalModelWeights extends ModelWeights {
  schema: typeof MODEL_ARTIFACT_SCHEMA;
  featureSchema: typeof MODEL_FEATURE_SCHEMA;
}

const DIMENSIONS = [
  ['fc1_w', 104, 12], ['fc1_b', 104],
  ['fc2_w', 64, 104], ['fc2_b', 64],
  ['fc3_w', 32, 64], ['fc3_b', 32],
  ['fc4_w', 1, 32], ['fc4_b', 1],
] as const;

let weights: ModelWeights | null = null;

function validNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

/** Validate exact architecture dimensions and reject NaN/Infinity weights. */
export function validateModelWeights(value: unknown): value is ModelWeights {
  if (!value || typeof value !== 'object') return false;
  const candidate = value as Record<string, unknown>;
  for (const [name, rows, columns] of DIMENSIONS) {
    const field = candidate[name];
    if (!Array.isArray(field) || field.length !== rows) return false;
    if (columns === undefined) {
      if (!field.every(validNumber)) return false;
      continue;
    }
    if (!field.every((row) => Array.isArray(row) && row.length === columns && row.every(validNumber))) return false;
  }
  return true;
}

/** Validate the versioned artifact consumed by the bundled scorer. */
export function validateModelArtifact(value: unknown): value is CanonicalModelWeights {
  if (!validateModelWeights(value)) return false;
  const candidate = value as Partial<CanonicalModelWeights>;
  return candidate.schema === MODEL_ARTIFACT_SCHEMA && candidate.featureSchema === MODEL_FEATURE_SCHEMA;
}

/** Load explicit model JSON through the Node adapter. This function never auto-loads. */
export async function loadJSONModel(modelPath: string): Promise<boolean> {
  try {
    const fs = await import('node:fs/promises');
    const text = await fs.readFile(modelPath, 'utf8');
    return loadJSONModelText(text);
  } catch {
    weights = null;
    return false;
  }
}

/** Browser-safe explicit loader for callers that provide JSON text themselves. */
export function loadJSONModelText(text: string): boolean {
  try {
    const parsed: unknown = JSON.parse(text);
    if (!validateModelArtifact(parsed)) {
      weights = null;
      return false;
    }
    weights = parsed;
    return true;
  } catch {
    weights = null;
    return false;
  }
}

export function isJSONModelAvailable(): boolean {
  return weights !== null;
}

function sigmoid(value: number): number {
  if (value >= 0) {
    const e = Math.exp(-value);
    return 1 / (1 + e);
  }
  const e = Math.exp(value);
  return e / (1 + e);
}

/** Run inference, returning null when no valid model is explicitly loaded. */
export function runJSONInference(features: ModelFeatureVector): number | null {
  if (!weights || features.length !== 12 || !features.every(validNumber)) return null;
  const { fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b, fc4_w, fc4_b } = weights;
  const relu = (value: number): number => value > 0 ? value : 0;
  const h1 = fc1_w.map((row, i) => relu(row.reduce((sum, weight, j) => sum + weight * features[j]!, fc1_b[i]!)));
  const h2 = fc2_w.map((row, i) => relu(row.reduce((sum, weight, j) => sum + weight * h1[j]!, fc2_b[i]!)));
  const h3 = fc3_w.map((row, i) => relu(row.reduce((sum, weight, j) => sum + weight * h2[j]!, fc3_b[i]!)));
  const output = sigmoid(fc4_w[0]!.reduce((sum, weight, j) => sum + weight * h3[j]!, fc4_b[0]!));
  return Number.isFinite(output) ? output : null;
}
