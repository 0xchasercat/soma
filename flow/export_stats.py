"""
Export flow normalization stats to JSON for the TypeScript runtime.

The ONNX graph carries weights but not the standardization constants, and the
TypeScript side cannot read a pickled .pt. This writes exactly the scalars
flow-synthesis.ts needs to map between model space and pixel space.
"""

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser(description="Export flow stats to JSON")
    p.add_argument("--flow", default="flow/flow.pt")
    p.add_argument("--dataset", default="flow/flow_dataset.pt")
    p.add_argument("--out", default="model/flow.stats.json")
    args = p.parse_args()

    ckpt = torch.load(args.flow, weights_only=False)
    stats = ckpt["stats"]
    layout = ckpt["layout"]
    arch = ckpt["arch"]
    dataset = torch.load(args.dataset, weights_only=False)
    if dataset["layout"] != layout:
        raise SystemExit("flow checkpoint and dataset layouts do not match")
    raw_x = dataset["x"] * stats["x_std"] + stats["x_mean"]
    n_free = layout["n_free"]
    terminal_magnitude = torch.hypot(
        raw_x[:, n_free - 1], raw_x[:, 2 * n_free - 1]
    )
    terminal_stop_epsilon = 1e-6
    terminal_stop_probability = float(
        (terminal_magnitude <= terminal_stop_epsilon).float().mean()
    )

    payload = {
        # v2 drops a constrained middle delta so the observed touchdown delta
        # remains explicit. v1 dropped the final delta and is incompatible.
        "schemaVersion": 2,
        "arch": {
            "dataDim": arch["data_dim"],
            "contextDim": arch["context_dim"],
            "nTransforms": arch["n_transforms"],
            "nBins": arch["n_bins"],
            "tailBound": arch["tail_bound"],
        },
        "layout": {
            "nSteps": layout["n_steps"],
            "nFree": layout["n_free"],
            "droppedDeltaIndex": layout["dropped_delta_index"],
            "duStart": layout["du"][0],
            "dvStart": layout["dv"][0],
            "logDurationIndex": layout["log_duration"],
        },
        "stats": {
            "xMean": stats["x_mean"].tolist(),
            "xStd": stats["x_std"].tolist(),
            "cMean": stats["c_mean"].tolist(),
            "cStd": stats["c_std"].tolist(),
        },
        "postprocess": {
            "terminalStopProbability": terminal_stop_probability,
            "terminalStopEpsilon": terminal_stop_epsilon,
        },
        "bestValNll": float(ckpt["best_val_nll"]),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
    print(f"  dataDim {payload['arch']['dataDim']}  logDurationIndex {payload['layout']['logDurationIndex']}")
    print(f"  terminalStopProbability {terminal_stop_probability:.6f}")


if __name__ == "__main__":
    main()
