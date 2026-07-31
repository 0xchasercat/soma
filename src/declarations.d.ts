declare const Bun: {
  file(path: string): { json(): Promise<unknown> };
  write(path: string, data: string): Promise<number>;
};

/**
 * Ambient type declaration for the optional peer dependency `onnxruntime-node`.
 *
 * The package is an optional peer dependency. Runtime loading is explicit and
 * failures are reported as unavailable; the scorer does not auto-load or claim
 * ONNX availability. This declaration keeps TypeScript independent of install
 * state.
 */
declare module 'onnxruntime-node' {
  interface Tensor {
    data: ArrayLike<number>;
  }

  interface RunResult {
    output?: Tensor;
    [key: string]: Tensor | undefined;
  }

  interface InferenceSession {
    readonly inputNames: readonly string[];
    readonly outputNames: readonly string[];
    run(feeds: Record<string, unknown>): Promise<RunResult>;
  }

  interface InferenceSessionFactory {
    create(path: string): Promise<InferenceSession>;
  }

  const InferenceSession: InferenceSessionFactory;

  class Tensor {
    constructor(type: string, data: ArrayLike<number>, dims: number[]);
    data: ArrayLike<number>;
  }
}
