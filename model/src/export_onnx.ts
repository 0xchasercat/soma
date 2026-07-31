/** Export canonical JSON MLP weights as a real ONNX protobuf graph. */

import { onnx } from 'onnx-proto';
import { validateModelArtifact, validateModelWeights, type ModelWeights } from '../../src/score/json_model.js';

const FLOAT = onnx.TensorProto.DataType.FLOAT;

function flattenTransposed(matrix: number[][]): number[] {
  const rows = matrix.length;
  const columns = matrix[0]?.length ?? 0;
  const output = new Array<number>(rows * columns);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      output[column * rows + row] = matrix[row]![column]!;
    }
  }
  return output;
}

function tensor(name: string, dims: number[], values: number[]): onnx.ITensorProto {
  return { name, dims, dataType: FLOAT, floatData: values };
}

function valueInfo(name: string, shape: number[]): onnx.IValueInfoProto {
  return {
    name,
    type: {
      tensorType: {
        elemType: FLOAT,
        shape: { dim: shape.map((dimValue) => ({ dimValue })) },
      },
    },
  };
}

function denseLayer(
  index: number,
  input: string,
  output: string,
  activation: 'Relu' | 'Sigmoid',
): onnx.INodeProto[] {
  const product = `fc${index}_product`;
  const biased = `fc${index}_biased`;
  return [
    { name: `fc${index}_matmul`, opType: 'MatMul', input: [input, `fc${index}_w`], output: [product] },
    { name: `fc${index}_add`, opType: 'Add', input: [product, `fc${index}_b`], output: [biased] },
    { name: `fc${index}_${activation.toLowerCase()}`, opType: activation, input: [biased], output: [output] },
  ];
}

export function encodeONNXModel(weights: ModelWeights): Uint8Array {
  if (!validateModelWeights(weights)) throw new Error('Cannot export invalid model weights');

  const graph: onnx.IGraphProto = {
    name: 'soma_motion_scorer',
    input: [valueInfo('input', [1, 12])],
    output: [valueInfo('output', [1, 1])],
    initializer: [
      tensor('fc1_w', [12, 104], flattenTransposed(weights.fc1_w)),
      tensor('fc1_b', [104], weights.fc1_b),
      tensor('fc2_w', [104, 64], flattenTransposed(weights.fc2_w)),
      tensor('fc2_b', [64], weights.fc2_b),
      tensor('fc3_w', [64, 32], flattenTransposed(weights.fc3_w)),
      tensor('fc3_b', [32], weights.fc3_b),
      tensor('fc4_w', [32, 1], flattenTransposed(weights.fc4_w)),
      tensor('fc4_b', [1], weights.fc4_b),
    ],
    node: [
      ...denseLayer(1, 'input', 'fc1_output', 'Relu'),
      ...denseLayer(2, 'fc1_output', 'fc2_output', 'Relu'),
      ...denseLayer(3, 'fc2_output', 'fc3_output', 'Relu'),
      ...denseLayer(4, 'fc3_output', 'output', 'Sigmoid'),
    ],
  };
  const model: onnx.IModelProto = {
    irVersion: onnx.Version.IR_VERSION,
    producerName: 'soma',
    producerVersion: '0.2.0',
    modelVersion: 1,
    docString: '12→104→64→32→1 behavioral trajectory scorer. Output is bot probability.',
    opsetImport: [{ domain: '', version: 13 }],
    graph,
    metadataProps: [
      { key: 'soma.feature_schema', value: 'soma.motion-features.v2' },
      { key: 'soma.output', value: 'bot_probability' },
    ],
  };
  const error = onnx.ModelProto.verify(model);
  if (error) throw new Error(`Invalid ONNX model graph: ${error}`);
  return onnx.ModelProto.encode(model).finish();
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  let inputPath = 'model/model.json';
  let outputPath = 'model/model.onnx';
  for (let index = 0; index < args.length; index++) {
    if (args[index] === '--input') inputPath = args[++index] ?? '';
    else if (args[index] === '--out') outputPath = args[++index] ?? '';
    else throw new Error(`Unknown argument: ${args[index]}`);
  }
  if (!inputPath || !outputPath) throw new Error('Usage: bun run model/src/export_onnx.ts [--input weights.json] [--out model.onnx]');

  const parsed: unknown = await Bun.file(inputPath).json();
  if (!validateModelArtifact(parsed)) throw new Error(`Invalid canonical JSON model: ${inputPath}`);
  const bytes = encodeONNXModel(parsed);
  await Bun.write(outputPath, bytes);
  console.log(`Exported ONNX model -> ${outputPath} (${bytes.byteLength} bytes)`);
}

if (import.meta.main) await main();
