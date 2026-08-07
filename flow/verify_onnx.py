"""Numerically verify the shipped ONNX inverse against its PyTorch checkpoint."""

import argparse

import numpy as np
import onnxruntime as ort
import torch

from train_flow import CONTEXT_DIM, DATA_DIM, build_flow


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify flow ONNX/PyTorch parity")
    parser.add_argument("--flow", default="flow/flow.pt")
    parser.add_argument("--onnx", default="model/flow.onnx")
    parser.add_argument("--dataset", default="flow/flow_dataset.pt")
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-error", type=float, default=2e-4)
    args = parser.parse_args()

    checkpoint = torch.load(args.flow, weights_only=False)
    if checkpoint["arch"]["data_dim"] != DATA_DIM:
        raise SystemExit("checkpoint data dimension does not match source architecture")
    if checkpoint["arch"]["context_dim"] != CONTEXT_DIM:
        raise SystemExit("checkpoint context dimension does not match source architecture")

    flow = build_flow()
    flow.load_state_dict(checkpoint["model_state"])
    flow.eval()
    dataset = torch.load(args.dataset, weights_only=False)
    contexts = dataset["c"]
    if contexts.shape[1] != CONTEXT_DIM:
        raise SystemExit("dataset context dimension does not match source architecture")

    generator = torch.Generator().manual_seed(args.seed)
    count = min(args.n, len(contexts))
    indices = torch.randperm(len(contexts), generator=generator)[:count]
    z = torch.randn(count, DATA_DIM, generator=generator)
    selected_contexts = contexts[indices]
    with torch.no_grad():
        expected, _ = flow._transform.inverse(z, context=selected_contexts)

    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    actual_rows = []
    for index in range(count):
        output = session.run(
            ["x"],
            {
                "z": z[index : index + 1].numpy().astype(np.float32),
                "context": selected_contexts[index : index + 1]
                .numpy()
                .astype(np.float32),
            },
        )[0]
        actual_rows.append(output[0])
    actual = np.stack(actual_rows)
    expected_np = expected.numpy()
    absolute = np.abs(actual - expected_np)
    max_error = float(absolute.max())
    mean_error = float(absolute.mean())
    finite = bool(np.isfinite(actual).all())
    print(
        f"ONNX parity: {count} samples, finite={finite}, "
        f"mean_abs_error={mean_error:.3e}, max_abs_error={max_error:.3e}"
    )
    if not finite or max_error > args.max_error:
        raise SystemExit(
            f"ONNX parity failed: max error {max_error:.3e} > {args.max_error:.3e}"
        )


if __name__ == "__main__":
    main()
