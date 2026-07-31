/**
 * Optional explicit ONNX inference with canonical schema/interface validation.
 * JSON remains the default portable format; ONNX uses an optional native peer.
 */
import type { ModelFeatureVector } from '../types.js';
/** Explicitly load a real ONNX model. Missing runtime or invalid files return false. */
export declare function loadONNXModel(modelPath: string): Promise<boolean>;
/**
 * Run inference on a feature vector.
 *
 * @param features 12-dimensional feature vector.
 * @returns Bot probability [0, 1], or null if model unavailable.
 */
export declare function runONNXInference(features: ModelFeatureVector): Promise<number | null>;
/**
 * Check if ONNX model is loaded and ready.
 */
export declare function isONNXAvailable(): boolean;
//# sourceMappingURL=onnx.d.ts.map