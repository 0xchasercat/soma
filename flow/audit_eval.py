"""
Phase 3 — Feature separability audit.

Reconstructs gestures from the trained flow and compares their feature
distributions against real captured humans.

Reconstruction is byte-for-byte the same procedure the TypeScript runtime uses
(see flow-synthesis.ts): denormalize -> restore constrained tail deltas ->
integrate to a path -> rescale by distance -> spread over the generated
duration. Auditing anything else would validate a pipeline that never ships.

Separability is reported as |AUC - 0.5| * 200:

    0%   -> the feature carries no information distinguishing gen from human
    100% -> the feature separates them perfectly

The parametric Bezier/AR(2) baseline scored ~91% on overshoot, ~87% on peak
timing and ~85% on bell-profile symmetry. Those are the numbers to beat.
"""

import argparse
import math

import numpy as np
import torch
from scipy.stats import mannwhitneyu

from train_flow import DATA_DIM, LOGDUR_IDX, N_FREE, build_flow

# ── Reconstruction ────────────────────────────────────────────────────────────


# Must stay identical to the constants in src/pointer/flow-synthesis.ts.
DECEL_WINDOW = 0.125
TERMINAL_SPEED_RATIO = 0.004
LATERAL_SMOOTH_PASSES = 2


def smooth_lateral(v: np.ndarray, passes: int = LATERAL_SMOOTH_PASSES) -> np.ndarray:
    """N-pass 3-tap binomial smoothing with pinned endpoints (smoothLateral in TS)."""
    if len(v) < 3 or passes < 1:
        return v
    cur = v
    for _ in range(passes):
        out = cur.copy()
        out[1:-1] = 0.25 * cur[:-2] + 0.5 * cur[1:-1] + 0.25 * cur[2:]
        cur = out
    return cur


def apply_decel_taper(t: np.ndarray) -> np.ndarray:
    """
    Stretch the final intervals in time so speed decays to TERMINAL_SPEED_RATIO
    of peak at touchdown, preserving total duration (applyDecelerationTaper).
    """
    n = len(t)
    total = float(t[-1])
    if n < 4 or total <= 0:
        return t
    start = 1.0 - DECEL_WINDOW
    phase = t[:-1] / total
    w = np.clip((phase - start) / DECEL_WINDOW, 0.0, 1.0)
    eased = w * w * (3 - 2 * w)  # smoothstep: continuous derivative at the edge
    stretch = 1.0 + eased * (1.0 / TERMINAL_SPEED_RATIO - 1.0)
    intervals = np.diff(t) * stretch
    s = intervals.sum()
    if s <= 0:
        return t
    out = np.concatenate([[0.0], np.cumsum(intervals * (total / s))])
    out[-1] = total
    return out


def reconstruct(
    x_std_vec: np.ndarray,
    dist_px: float,
    x_mean: np.ndarray,
    x_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardized N-dim model vector -> (u_px, v_px, t_ms).

    Mirrors reconstructGesture() in src/pointer/flow-synthesis.ts exactly:
    no taper, no smoothing -- the flow output is used as-is.
    """
    raw = x_std_vec * x_std + x_mean

    du = raw[:N_FREE]
    dv = raw[N_FREE : 2 * N_FREE]
    duration_ms = float(math.exp(raw[LOGDUR_IDX]))

    du = np.concatenate([du, [1.0 - du.sum()]])
    dv = np.concatenate([dv, [0.0 - dv.sum()]])

    u = np.concatenate([[0.0], np.cumsum(du)]) * dist_px
    v = np.concatenate([[0.0], np.cumsum(dv)]) * dist_px
    t = np.linspace(0.0, duration_ms, len(u))
    return u, v, t


# ── Features ──────────────────────────────────────────────────────────────────


def features(u: np.ndarray, v: np.ndarray, t: np.ndarray) -> dict[str, float]:
    """
    Pixel-space gesture features matching the discriminator's inputs.

    u is the on-axis coordinate (0 -> distance), v the lateral deviation.
    """
    du, dv = np.diff(u), np.diff(v)
    dt = np.diff(t)
    dt = np.where(dt <= 0, 1e-6, dt)

    step = np.hypot(du, dv)
    speed = step / dt
    peak_i = int(speed.argmax())
    peak_speed = float(speed[peak_i])
    total_t = float(t[-1] - t[0])
    dist = float(u[-1] - u[0])

    # Overshoot: furthest travel past the endpoint along the approach axis.
    overshoot_px = float(max(0.0, u.max() - u[-1]))

    # Bell profile: single dominant acceleration hump. Measured as the peak
    # sitting in the middle third of elapsed time with a unimodal speed trace.
    peak_time_frac = float(t[peak_i + 1] / total_t) if total_t > 0 else 0.0
    sign_flips = int(np.sum(np.diff(np.sign(np.diff(speed))) != 0))
    is_bell = float(0.25 <= peak_time_frac <= 0.75 and sign_flips <= 4)

    # Submovements: local speed minima below 60% of peak, the classic
    # intermittent-control correction signature.
    submovements = 0
    for i in range(1, len(speed) - 1):
        if speed[i] < speed[i - 1] and speed[i] < speed[i + 1] and speed[i] < 0.6 * peak_speed:
            submovements += 1

    path_len = float(step.sum())
    straight = math.hypot(u[-1] - u[0], v[-1] - v[0])
    curvature = path_len / straight - 1.0 if straight > 1e-9 else 0.0

    # Tremor: fraction of lateral-jerk energy in the upper frequency band.
    if len(dv) >= 16:
        mag = np.abs(np.fft.rfft(dv - dv.mean())) ** 2
        freqs = np.fft.rfftfreq(len(dv))
        tremor = float(mag[freqs > 0.3].sum() / (mag.sum() + 1e-12))
    else:
        tremor = 0.0

    # Terminal deceleration: how much speed remains at touchdown.
    final_ratio = float(speed[-1] / peak_speed) if peak_speed > 0 else 0.0

    return {
        "overshoot_frac": overshoot_px / dist if dist > 1e-9 else 0.0,
        "has_overshoot": float(overshoot_px > 0.01 * dist),
        "peak_time_frac": peak_time_frac,
        "bell_profile": is_bell,
        "submovement_count": float(submovements),
        "curvature": curvature,
        "tremor_energy": tremor,
        "final_speed_ratio": final_ratio,
        "duration_ms": total_t,
        "peak_speed_px_s": peak_speed * 1000.0,
        "max_lateral_frac": float(np.abs(v).max() / dist) if dist > 1e-9 else 0.0,
    }


FEATURE_ORDER = [
    "has_overshoot",
    "overshoot_frac",
    "peak_time_frac",
    "bell_profile",
    "submovement_count",
    "curvature",
    "tremor_energy",
    "final_speed_ratio",
    "max_lateral_frac",
    "duration_ms",
    "peak_speed_px_s",
]


def separability(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return (AUC, separability_pct). Constant-everywhere features score 0."""
    if len(a) == 0 or len(b) == 0:
        return 0.5, 0.0
    if np.allclose(a, a[0]) and np.allclose(b, b[0]):
        return (0.5, 0.0) if math.isclose(float(a[0]), float(b[0])) else (1.0, 100.0)
    stat, _ = mannwhitneyu(a, b, alternative="two-sided")
    auc = stat / (len(a) * len(b))
    return auc, abs(auc - 0.5) * 200.0


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description="Audit flow-generated gesture realism")
    p.add_argument("--flow", default="flow/flow.pt")
    p.add_argument("--dataset", default="flow/flow_dataset.pt")
    p.add_argument("--n", type=int, default=3000, help="samples per side")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    ckpt = torch.load(args.flow, weights_only=False)
    flow = build_flow()
    flow.load_state_dict(ckpt["model_state"])
    flow.eval()

    data = torch.load(args.dataset, weights_only=False)
    x_all = data["x"].numpy()
    c_all = data["c"]
    meta_all = data["meta"].numpy()  # [distance_px, target_size_px, duration_ms]
    stats = ckpt["stats"]
    x_mean = stats["x_mean"].numpy()
    x_std = stats["x_std"].numpy()

    n = min(args.n, len(x_all))
    idx = rng.choice(len(x_all), n, replace=False)

    print(f"flow      : {args.flow}  (val NLL {ckpt['best_val_nll']:.1f})")
    print(f"comparing : {n} real vs {n} generated\n")

    # Real gestures, reconstructed through the identical path so the comparison
    # isolates the model rather than the reconstruction.
    real_feats = [
        features(*reconstruct(x_all[i], float(meta_all[i, 0]), x_mean, x_std))
        for i in idx
    ]

    # Generated gestures conditioned on the same (distance, target_size) pairs,
    # so any difference is the trajectory shape, not the context distribution.
    with torch.no_grad():
        z = torch.randn(n, DATA_DIM)
        xs, _ = flow._transform.inverse(z, context=c_all[idx])
    xs = xs.numpy()

    n_nonfinite = int((~np.isfinite(xs).all(axis=1)).sum())
    gen_feats = [
        features(*reconstruct(xs[k], float(meta_all[i, 0]), x_mean, x_std))
        for k, i in enumerate(idx)
        if np.isfinite(xs[k]).all()
    ]
    if n_nonfinite:
        print(f"WARNING: {n_nonfinite} non-finite samples dropped\n")

    print(f"{'feature':<20} {'real':>11} {'generated':>11} {'AUC':>7} {'sep':>7}")
    print("-" * 60)
    seps = {}
    for name in FEATURE_ORDER:
        r = np.array([f[name] for f in real_feats])
        g = np.array([f[name] for f in gen_feats])
        auc, sep = separability(r, g)
        seps[name] = sep
        flag = "  <-- " if sep > 30 else ""
        print(f"{name:<20} {r.mean():>11.4f} {g.mean():>11.4f} {auc:>7.3f} {sep:>6.1f}%{flag}")
    print("-" * 60)

    headline = ["has_overshoot", "peak_time_frac", "bell_profile"]
    print("\nheadline features vs parametric baseline:")
    for name, base in zip(headline, (91.1, 87.3, 84.7)):
        now = seps[name]
        delta = base - now
        print(f"  {name:<18} {base:>5.1f}% -> {now:>5.1f}%   ({delta:+.1f} pts)")

    mean_sep = float(np.mean(list(seps.values())))
    print(f"\nmean separability across {len(seps)} features: {mean_sep:.1f}%")
    print("0% = indistinguishable; the parametric baseline averaged well above 80%")


if __name__ == "__main__":
    main()
