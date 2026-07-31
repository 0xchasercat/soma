/**
 * Canonical model feature extractor.
 *
 * Training imports the runtime implementation directly so the model cannot be
 * trained against a subtly different feature definition than inference uses.
 */
export { extractFeatureVector } from '../../src/score/heuristic.js';
