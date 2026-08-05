"""
Phase 2 — Conditional normalizing flow over pointer gestures.

Model
-----
Density p(x | c) over the 255-d gesture vector produced by dataset_prep.py:

    x = [ du[0..126] (127), dv[0..126] (127), log(duration_ms) (1) ]
    c = [ log(distance_px), log(target_size_px) ]

Trained by exact negative log-likelihood:  L = -log p(x | c).

Coupling masks split by TIME INDEX, not flat position
-----------------------------------------------------
The naive flat half-split would place every du in one set and every dv in the
other, so a coupling conditioner would never see both axes of the same
timestep -- it could not learn that lateral deviation correlates with forward
speed, which is precisely the structure a 1D-CNN is here to capture.

Instead layer i leaves the checkerboard timestep set

    S_i = { t : (t + i) mod 2 == 0 }

unchanged, alongside log-duration on even layers. nflows treats mask <= 0 as
the identity set and hands those features to the conditioner in ascending
index order, so the conditioner input is exactly

    du[S_i] ++ dv[S_i] ( ++ log-duration if i even )

which reshapes to (B, 2, |S_i|) for convolution across time. Alternating i
guarantees every coordinate is transformed by half the layers.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from nflows.distributions import StandardNormal
from nflows.flows import Flow
from nflows.transforms import (
    ActNorm,
    CompositeTransform,
    PiecewiseRationalQuadraticCouplingTransform,
)

# ── Architecture ──────────────────────────────────────────────────────────────

N_FREE = 79    # free spatial deltas per axis (N_STEPS=80 minus constrained tail)
DATA_DIM = 2 * N_FREE + 1  # 159
CONTEXT_DIM = 2
LOGDUR_IDX = 2 * N_FREE  # index of log(duration_ms)
N_TRANSFORMS = 6
N_BINS = 8
TAIL_BOUND = 6.0  # standardized data is unit-variance; 6 sigma covers the corpus


class CNN1DConditioner(nn.Module):
    """
    Coupling conditioner: 1D-CNN over time with FiLM context injection.

    Input  (B, in_features) = du[S] ++ dv[S] ( ++ log-duration )
    Output (B, out_features) = RQ-spline parameters for the transformed set
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_time: int,
        has_logdur: bool,
        context_features: int = CONTEXT_DIM,
        width: int = 96,
    ):
        super().__init__()
        expected = 2 * n_time + (1 if has_logdur else 0)
        if in_features != expected:
            raise ValueError(
                f"conditioner expected {expected} identity features "
                f"(2x{n_time} deltas + {int(has_logdur)} logdur), got {in_features}"
            )
        self.n_time = n_time
        self.has_logdur = has_logdur

        ctx_hidden = 64
        self.ctx_embed = nn.Sequential(
            nn.Linear(context_features, ctx_hidden),
            nn.SiLU(),
            nn.Linear(ctx_hidden, ctx_hidden),
            nn.SiLU(),
        )
        # FiLM modulates the two spatial channels from the context.
        self.film = nn.Linear(ctx_hidden, 2 * 2)

        self.conv_net = nn.Sequential(
            nn.Conv1d(2, width, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(width, width, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(width, width, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        # Global average pool: always ONNX-exportable regardless of input length
        # (AdaptiveAvgPool1d(4) fails ONNX when input length is not divisible by 4).
        self.pool = nn.AdaptiveAvgPool1d(1)
        head_in = width * 1 + ctx_hidden + (1 if has_logdur else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.SiLU(),
            nn.Linear(256, out_features),
        )
        # Zero-init the final layer: the coupling starts as identity, which
        # keeps the first NLL evaluations finite.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        B = x.shape[0]
        if self.has_logdur:
            deltas, logdur = x[:, :-1], x[:, -1:]
        else:
            deltas, logdur = x, None

        # (B, 2*n_time) -> (B, 2, n_time): du row then dv row, matching the
        # ascending-index order nflows uses for identity features.
        h = deltas.view(B, 2, self.n_time)

        if context is not None:
            ctx = self.ctx_embed(context)
            film = self.film(ctx)
            gamma = film[:, :2].unsqueeze(-1)
            beta = film[:, 2:].unsqueeze(-1)
            h = h * (1.0 + gamma) + beta
        else:
            ctx = torch.zeros(B, self.ctx_embed[-2].out_features, device=x.device)

        h = self.conv_net(h)
        h = self.pool(h).flatten(1)
        parts = [h, ctx]
        if logdur is not None:
            parts.append(logdur)
        return self.head(torch.cat(parts, dim=1))


def build_mask(layer: int) -> torch.Tensor:
    """
    Checkerboard-in-time mask, in nflows polarity.

    mask <= 0 -> identity  (passed unchanged, fed to the conditioner)
    mask >  0 -> transformed by the spline

    Even layers hold log-duration in the identity set so the conditioner can
    condition spatial splines on total gesture time; odd layers transform it.
    """
    mask = torch.ones(DATA_DIM)
    for t in range(N_FREE):
        if (t + layer) % 2 == 0:
            mask[t] = 0  # du[t] identity
            mask[N_FREE + t] = 0  # dv[t] identity
    mask[LOGDUR_IDX] = 0.0 if layer % 2 == 0 else 1.0
    return mask


def build_flow() -> Flow:
    transforms: list = []
    for i in range(N_TRANSFORMS):
        mask = build_mask(i)
        # Derive the conditioner's input shape from the mask itself so the two
        # definitions cannot drift apart.
        identity = mask <= 0
        n_time = int(identity[:N_FREE].sum())
        has_logdur = bool(identity[LOGDUR_IDX])
        if int(identity[N_FREE : 2 * N_FREE].sum()) != n_time:
            raise ValueError(f"layer {i}: du/dv identity sets must be symmetric")

        def make_net(in_f: int, out_f: int, _n=n_time, _h=has_logdur) -> nn.Module:
            return CNN1DConditioner(in_f, out_f, n_time=_n, has_logdur=_h)

        transforms.append(ActNorm(features=DATA_DIM))
        transforms.append(
            PiecewiseRationalQuadraticCouplingTransform(
                mask=mask,
                transform_net_create_fn=make_net,
                num_bins=N_BINS,
                tails="linear",
                tail_bound=TAIL_BOUND,
            )
        )
    return Flow(
        transform=CompositeTransform(transforms),
        distribution=StandardNormal(shape=[DATA_DIM]),
    )


# ── Training ──────────────────────────────────────────────────────────────────


def train(args) -> None:
    print(f"Loading dataset from {args.dataset} ...")
    data = torch.load(args.dataset, weights_only=False)
    x_all, c_all, stats = data["x"], data["c"], data["stats"]
    if x_all.shape[1] != DATA_DIM:
        raise SystemExit(
            f"dataset dim {x_all.shape[1]} != model dim {DATA_DIM}; re-run dataset_prep.py"
        )
    n = x_all.shape[0]
    print(f"  N = {n:,}   x {tuple(x_all.shape)}   c {tuple(c_all.shape)}")

    n_val = max(1, int(n * 0.1))
    train_ds, val_ds = random_split(
        TensorDataset(x_all, c_all),
        [n - n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    # MPS is the GPU on Apple Silicon. nflows registers one float64 buffer
    # (_distribution._log_z, the base Gaussian log-normalizer) which MPS
    # cannot hold. It is a scalar constant (log(2π)/2); casting to float32
    # loses ~1e-7 relative error, negligible against NLL training noise.
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    flow = build_flow()
    # Cast before .to(device): _log_z is float64 but MPS only holds float32.
    # Do this on CPU first so .to(device) never sees a float64 buffer.
    for module in flow.modules():
        if hasattr(module, "_log_z") and module._log_z.dtype == torch.float64:
            module._log_z = module._log_z.to(torch.float32)
    flow = flow.to(device)
    n_params = sum(p.numel() for p in flow.parameters() if p.requires_grad)
    print(f"  device {device}   train {n - n_val:,}   val {n_val:,}   params {n_params:,}")
    print(f"  {N_TRANSFORMS} coupling layers, {N_BINS} bins, tail_bound {TAIL_BOUND}")

    opt = optim.AdamW(flow.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    best_val = math.inf
    best_state: dict | None = None

    for epoch in range(1, args.epochs + 1):
        flow.train()
        tot, seen = 0.0, 0
        for xb, cb in train_loader:
            xb, cb = xb.to(device), cb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = -flow.log_prob(xb, context=cb).mean()
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at epoch {epoch} -- training diverged")
            loss.backward()
            nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            opt.step()
            tot += loss.item() * xb.shape[0]
            seen += xb.shape[0]
        sched.step()
        train_nll = tot / seen

        flow.eval()
        tot, seen = 0.0, 0
        with torch.no_grad():
            for xb, cb in val_loader:
                xb, cb = xb.to(device), cb.to(device)
                tot += (-flow.log_prob(xb, context=cb).mean()).item() * xb.shape[0]
                seen += xb.shape[0]
        val_nll = tot / seen

        if val_nll < best_val:
            best_val = val_nll
            best_state = {k: v.detach().cpu().clone() for k, v in flow.state_dict().items()}

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            mark = " *" if val_nll == best_val else ""
            print(
                f"  epoch {epoch:3d}/{args.epochs}  train_nll={train_nll:9.3f}  "
                f"val_nll={val_nll:9.3f}  lr={sched.get_last_lr()[0]:.2e}{mark}"
            )

    if best_state is None:
        raise SystemExit("no epoch completed")
    print(f"\nbest val NLL {best_val:.3f} nats over {DATA_DIM} dims "
          f"({best_val / DATA_DIM:.4f} per dim)")

    flow.load_state_dict(best_state)
    flow.to("cpu").eval()

    # Sample-quality smoke test: the endpoint constraint is restored
    # analytically, so what matters is that samples are finite and that
    # duration lands in a plausible range.
    with torch.no_grad():
        c_probe = data["c"][: min(512, n)]
        z = torch.randn(c_probe.shape[0], DATA_DIM)
        xs, _ = flow._transform.inverse(z, context=c_probe)
    finite = torch.isfinite(xs).all(dim=1)
    x_mean, x_std = stats["x_mean"], stats["x_std"]
    dur_ms = torch.exp(xs[:, LOGDUR_IDX] * x_std[LOGDUR_IDX] + x_mean[LOGDUR_IDX])
    print(f"sample check: {int(finite.sum())}/{len(xs)} finite   "
          f"duration p5/p50/p95 = {dur_ms.quantile(0.05):.0f} / "
          f"{dur_ms.median():.0f} / {dur_ms.quantile(0.95):.0f} ms")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": flow.state_dict(),
            "stats": stats,
            "layout": data["layout"],
            "arch": {
                "data_dim": DATA_DIM,
                "context_dim": CONTEXT_DIM,
                "n_transforms": N_TRANSFORMS,
                "n_bins": N_BINS,
                "tail_bound": TAIL_BOUND,
            },
            "best_val_nll": best_val,
        },
        out,
    )
    print(f"saved {out}")
    onnx_path = Path(args.onnx_out)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    export_onnx(flow, onnx_path)
    print(f"saved {onnx_path}")


class _FlowInverse(nn.Module):
    """Thin wrapper exporting the inverse pass (z -> x) to ONNX."""

    def __init__(self, flow: Flow):
        super().__init__()
        self.flow = flow

    def forward(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x, _ = self.flow._transform.inverse(z, context=context)
        return x


def export_onnx(flow: Flow, path: Path) -> None:
    """Export the inverse pass (z -> x) to ONNX.

    PyTorch >= 2.6 routes torch.onnx.export through the dynamo exporter, which
    cannot guard on the data-dependent ``torch.any(inside_interval_mask)`` branch
    inside the nflows RQ-spline, and rejects a pre-traced ScriptModule outright.

    ``fallback=True`` makes the exporter retry with the legacy TorchScript path
    (``torch.onnx.utils.export``) when the dynamo attempt fails.  That path
    traces the module internally, folding the data-dependent branches into the
    concrete path taken for z ~ N(0,1) -- which is the only path the runtime
    ever exercises, since every sampled value lands inside the spline tails.

    Batch size is fixed at 1; the TypeScript runtime calls the model one gesture
    at a time.
    """
    wrapper = _FlowInverse(flow).eval()
    z_dummy = torch.randn(1, DATA_DIM, dtype=torch.float32)
    c_dummy = torch.zeros(1, CONTEXT_DIM, dtype=torch.float32)
    with torch.no_grad():
        _ = wrapper(z_dummy, c_dummy)  # warm up
    torch.onnx.export(
        wrapper,
        (z_dummy, c_dummy),
        str(path),
        input_names=["z", "context"],
        output_names=["x"],
        opset_version=17,
        fallback=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Train the conditional gesture flow")
    p.add_argument("--dataset", default="flow/flow_dataset.pt")
    p.add_argument("--out", default="flow/flow.pt",
                   help="Checkpoint path (.pt); training artifact, not shipped.")
    p.add_argument("--onnx-out", default="model/flow.onnx",
                   help="ONNX export path; must match FLOW_ONNX_URL in src/index.ts.")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=1024,
                   help="Batch size. 1024 gives ~10x speedup on MPS vs 256.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
