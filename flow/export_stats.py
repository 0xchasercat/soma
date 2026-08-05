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
    p.add_argument("--out", default="model/flow.stats.json")
    args = p.parse_args()

    ckpt = torch.load(args.flow, weights_only=False)
    stats = ckpt["stats"]
    layout = ckpt["layout"]
    arch = ckpt["arch"]

    payload = {
        "schemaVersion": 1,
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
        "bestValNll": float(ckpt["best_val_nll"]),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
    print(f"  dataDim {payload['arch']['dataDim']}  logDurationIndex {payload['layout']['logDurationIndex']}")


if __name__ == "__main__":
    main()
