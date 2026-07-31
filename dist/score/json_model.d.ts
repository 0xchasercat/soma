/** Explicit JSON model loader and inference. JSON is the canonical model format. */
import type { ModelFeatureVector } from '../types.js';
export interface ModelWeights {
    fc1_w: number[][];
    fc1_b: number[];
    fc2_w: number[][];
    fc2_b: number[];
    fc3_w: number[][];
    fc3_b: number[];
    fc4_w: number[][];
    fc4_b: number[];
}
export declare const MODEL_ARTIFACT_SCHEMA = "soma.motion-model.v1";
export declare const MODEL_FEATURE_SCHEMA = "soma.motion-features.v2";
export interface CanonicalModelWeights extends ModelWeights {
    schema: typeof MODEL_ARTIFACT_SCHEMA;
    featureSchema: typeof MODEL_FEATURE_SCHEMA;
}
/** Validate exact architecture dimensions and reject NaN/Infinity weights. */
export declare function validateModelWeights(value: unknown): value is ModelWeights;
/** Validate the versioned artifact consumed by the bundled scorer. */
export declare function validateModelArtifact(value: unknown): value is CanonicalModelWeights;
/** Load explicit model JSON through the Node adapter. This function never auto-loads. */
export declare function loadJSONModel(modelPath: string): Promise<boolean>;
/** Browser-safe explicit loader for callers that provide JSON text themselves. */
export declare function loadJSONModelText(text: string): boolean;
export declare function isJSONModelAvailable(): boolean;
/** Run inference, returning null when no valid model is explicitly loaded. */
export declare function runJSONInference(features: ModelFeatureVector): number | null;
//# sourceMappingURL=json_model.d.ts.map