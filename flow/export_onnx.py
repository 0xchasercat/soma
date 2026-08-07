"""Re-export a trained flow checkpoint without retraining it."""

import argparse
from pathlib import Path

import torch

from train_flow import build_flow, export_onnx


def main() -> None:
    parser = argparse.ArgumentParser(description="Export flow checkpoint to ONNX")
    parser.add_argument("--flow", default="flow/flow.pt")
    parser.add_argument("--out", default="model/flow.onnx")
    args = parser.parse_args()
    checkpoint = torch.load(args.flow, weights_only=False)
    flow = build_flow()
    flow.load_state_dict(checkpoint["model_state"])
    flow.eval()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    export_onnx(flow, output)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
