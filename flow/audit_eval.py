"""
Phase 3 — Feature separability audit.

Reconstructs gestures from the trained flow and compares their feature
distributions against real captured humans.

Reconstruction mirrors the TypeScript runtime (see flow-synthesis.ts):
denormalize -> restore the constrained middle deltas -> integrate -> rescale by
distance -> spread over the generated duration -> resample for dispatch. The
report compares both the raw flow and the shipped default AR(2) tremor path so a
post-processing regression cannot hide behind a good raw model score.

Separability is reported as |AUC - 0.5| * 200:

    0%   -> the feature carries no information distinguishing gen from human
    100% -> the feature separates them perfectly

The report is intentionally self-contained. Historical hard-coded baseline
numbers are not mixed into current results because they were produced by a
different vector layout and an earlier, now-superseded binary bell metric.
"""

import argparse
import math

import numpy as np
import torch
from scipy.stats import mannwhitneyu

from dataset_prep import DROPPED_DELTA_INDEX, N_STEPS
from train_flow import DATA_DIM, LOGDUR_IDX, N_FREE, build_flow

# ── Reconstruction ────────────────────────────────────────────────────────────


# Must stay identical to src/pointer/flow-synthesis.ts and pointer/tremor.ts.
FRAME_MS = 1000.0 / 60.0
MAX_DISPATCH_FRAMES = 4096
AR2_PHI1 = 1.7905
AR2_PHI2 = -0.9604
AR2_STATIONARY_VARIANCE = (1 - AR2_PHI2) / (
    (1 + AR2_PHI2)
    * (1 - AR2_PHI1 - AR2_PHI2)
    * (1 + AR2_PHI1 - AR2_PHI2)
)
AR2_INNOVATION_SCALE = 1.0 / math.sqrt(AR2_STATIONARY_VARIANCE)


def reconstruct(
    x_std_vec: np.ndarray,
    dist_px: float,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    force_terminal_stop: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardized N-dim model vector -> (u_px, v_px, t_ms).

    Mirrors reconstructGesture() in src/pointer/flow-synthesis.ts exactly:
    no taper, no smoothing -- the flow output is used as-is.
    """
    raw = x_std_vec * x_std + x_mean

    du_free = raw[:N_FREE]
    dv_free = raw[N_FREE : 2 * N_FREE]
    if force_terminal_stop:
        du_free = du_free.copy()
        dv_free = dv_free.copy()
        du_free[-1] = 0.0
        dv_free[-1] = 0.0
    duration_ms = float(math.exp(raw[LOGDUR_IDX]))

    du = np.insert(du_free, DROPPED_DELTA_INDEX, 1.0 - du_free.sum())
    dv = np.insert(dv_free, DROPPED_DELTA_INDEX, 0.0 - dv_free.sum())
    assert len(du) == N_STEPS and len(dv) == N_STEPS

    u = np.concatenate([[0.0], np.cumsum(du)]) * dist_px
    v = np.concatenate([[0.0], np.cumsum(dv)]) * dist_px
    t = np.linspace(0.0, duration_ms, len(u))
    return u, v, t


def resample_for_dispatch(
    u: np.ndarray, v: np.ndarray, t: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of resampleToFrameRate, including its long-duration frame cap."""
    total = float(t[-1])
    # JavaScript Math.round for non-negative elapsed time (Python round uses
    # ties-to-even and can differ by one frame at exact half intervals).
    desired = math.floor(total / FRAME_MS + 0.5) + 1
    frames = max(2, min(MAX_DISPATCH_FRAMES, desired))
    target = np.linspace(0.0, total, frames)
    return np.interp(target, t, u), np.interp(target, t, v), target


def pin_dispatch_terminal_stop(
    path: tuple[np.ndarray, np.ndarray, np.ndarray], enabled: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reserve the last 60 Hz dispatch interval for an exact stationary state."""
    if not enabled or len(path[0]) <= 2:
        return path
    u, v, t = path
    u, v = u.copy(), v.copy()
    u[-2], v[-2] = u[-1], v[-1]
    return u, v, t


def add_default_tremor(
    u: np.ndarray,
    v: np.ndarray,
    t: np.ndarray,
    amplitude: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distribution-equivalent port of addTremor with exact endpoint pinning."""
    if len(u) < 2 or amplitude <= 0:
        return u, v, t
    out_u, out_v = u.copy(), v.copy()
    x_prev_1, x_prev_2, y_prev_1, y_prev_2 = rng.normal(0, amplitude, 4)
    for index in range(1, len(u) - 1):
        dt = float(t[index] - t[index - 1])
        if not math.isfinite(dt) or dt <= 0:
            dt = FRAME_MS
        rate_scale = math.sqrt(min(4.0, max(0.25, dt / FRAME_MS)))
        innovation_std = amplitude * AR2_INNOVATION_SCALE * rate_scale
        x_noise = (
            AR2_PHI1 * x_prev_1
            + AR2_PHI2 * x_prev_2
            + rng.normal(0, innovation_std)
        )
        y_noise = (
            AR2_PHI1 * y_prev_1
            + AR2_PHI2 * y_prev_2
            + rng.normal(0, innovation_std)
        )
        x_prev_2, x_prev_1 = x_prev_1, x_noise
        y_prev_2, y_prev_1 = y_prev_1, y_noise
        out_u[index] += x_noise
        out_v[index] += y_noise
    return out_u, out_v, t


# ── Features ──────────────────────────────────────────────────────────────────


def sigma_lognormal_fit(speeds: np.ndarray) -> float:
    """Port of src/measure/velocity.ts fitSigmaLognormal for audit parity."""
    finite = np.maximum(0.0, np.nan_to_num(speeds, nan=0.0, posinf=0.0, neginf=0.0))
    if len(finite) < 5:
        return 0.0
    sample_count = min(64, len(finite))
    positions = np.linspace(0.0, len(finite) - 1, sample_count)
    y = np.interp(positions, np.arange(len(finite)), finite)
    max_speed = float(y.max())
    if max_speed <= 1e-6:
        return 0.0
    normalized = y / max_speed
    n = len(normalized)
    peak = int(normalized.argmax()) / max(1, n - 1)
    local_peaks = [
        i / (n - 1)
        for i in range(1, n - 1)
        if normalized[i] >= normalized[i - 1] and normalized[i] >= normalized[i + 1]
    ]
    local_peaks.sort(key=lambda center: normalized[round(center * (n - 1))], reverse=True)
    centers: list[float] = []
    for center in [peak, *local_peaks]:
        if 0.02 < center < 0.99 and center not in centers:
            centers.append(center)
        if len(centers) == 3:
            break
    if not centers:
        centers.append(max(0.05, min(0.95, peak)))

    best_r2 = 0.0
    for count in range(1, len(centers) + 1):
        selected = centers[:count]
        for sigma in (0.25, 0.4, 0.65, 0.85):
            basis = []
            for center in selected:
                mu = math.log(max(0.02, center)) + sigma * sigma
                time = (np.arange(n) + 0.5) / n
                z = (np.log(time) - mu) / sigma
                basis.append(np.exp(-0.5 * z * z) / (time * sigma))
            coefficients = np.zeros(len(basis))
            for _ in range(5):
                for k, lobe in enumerate(basis):
                    prediction = sum(coefficients[j] * basis[j] for j in range(len(basis)))
                    residual_dot = float(np.dot(lobe, normalized - prediction + coefficients[k] * lobe))
                    norm = float(np.dot(lobe, lobe))
                    coefficients[k] = max(0.0, residual_dot / norm) if norm > 1e-12 else 0.0
            predicted = sum(coefficients[j] * basis[j] for j in range(len(basis)))
            pred_max = float(predicted.max())
            if pred_max <= 1e-12:
                continue
            predicted = predicted / pred_max
            total = float(np.square(normalized - normalized.mean()).sum())
            if total > 0:
                r2 = 1.0 - float(np.square(normalized - predicted).sum()) / total
                best_r2 = max(best_r2, r2)
    return max(0.0, min(1.0, best_r2))


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

    peak_time_frac = float(t[peak_i + 1] / total_t) if total_t > 0 else 0.0
    sigma_fit = sigma_lognormal_fit(speed)

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
        "sigma_lognormal_fit": sigma_fit,
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
    "sigma_lognormal_fit",
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


def report(
    title: str,
    real_feats: list[dict[str, float]],
    generated_feats: list[dict[str, float]],
) -> float:
    print(f"\n{title}")
    print(f"{'feature':<20} {'real':>11} {'generated':>11} {'AUC':>7} {'sep':>7}")
    print("-" * 60)
    seps: dict[str, float] = {}
    for name in FEATURE_ORDER:
        real = np.array([feature[name] for feature in real_feats])
        generated = np.array([feature[name] for feature in generated_feats])
        auc, sep = separability(real, generated)
        seps[name] = sep
        flag = "  <--" if sep > 30 else ""
        print(
            f"{name:<20} {real.mean():>11.4f} {generated.mean():>11.4f} "
            f"{auc:>7.3f} {sep:>6.1f}%{flag}"
        )
    print("-" * 60)
    mean_sep = float(np.mean(list(seps.values())))
    print(f"mean separability across {len(seps)} features: {mean_sep:.1f}%")
    return mean_sep


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description="Audit flow-generated gesture realism")
    p.add_argument("--flow", default="flow/flow.pt")
    p.add_argument("--dataset", default="flow/flow_dataset.pt")
    p.add_argument("--n", type=int, default=3000, help="samples per side")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--tremor-amplitude",
        type=float,
        default=0.0,
        help="optional legacy AR(2) tremor amplitude; shipped default is zero",
    )
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
    raw_x_all = x_all * x_std + x_mean
    terminal_magnitude = np.hypot(
        raw_x_all[:, N_FREE - 1], raw_x_all[:, 2 * N_FREE - 1]
    )
    measured_stop_probability = float(np.mean(terminal_magnitude <= 1e-6))

    saved_validation = ckpt.get("validation_indices")
    if isinstance(saved_validation, torch.Tensor):
        candidates = saved_validation.numpy().astype(np.int64)
        candidates = candidates[(candidates >= 0) & (candidates < len(x_all))]
        cohort = ckpt.get("split", {}).get("kind", "saved validation")
    else:
        candidates = np.arange(len(x_all), dtype=np.int64)
        cohort = "full dataset (legacy checkpoint)"
    n = min(args.n, len(candidates))
    idx = rng.choice(candidates, n, replace=False)

    print(f"flow      : {args.flow}  (val NLL {ckpt['best_val_nll']:.1f})")
    print(f"comparing : {n} real vs {n} generated ({cohort})\n")
    print(f"terminal stop point mass: {measured_stop_probability:.4f}")

    # Real gestures, reconstructed through the identical path so the comparison
    # isolates the model rather than the reconstruction.
    stop_flags = meta_all[idx, 3] > 0.5
    real_paths = [
        pin_dispatch_terminal_stop(
            resample_for_dispatch(
                *reconstruct(
                    x_all[i],
                    float(meta_all[i, 0]),
                    x_mean,
                    x_std,
                    force_terminal_stop=bool(stop_flags[k]),
                )
            ),
            bool(stop_flags[k]),
        )
        for k, i in enumerate(idx)
    ]
    real_feats = [features(*path) for path in real_paths]

    # Generated gestures conditioned on the same (distance, target_size) pairs,
    # so any difference is the trajectory shape, not the context distribution.
    with torch.no_grad():
        z = torch.randn(n, DATA_DIM)
        xs, _ = flow._transform.inverse(z, context=c_all[idx])
    xs = xs.numpy()

    n_nonfinite = int((~np.isfinite(xs).all(axis=1)).sum())
    generated_paths = [
        resample_for_dispatch(
            *reconstruct(xs[k], float(meta_all[i, 0]), x_mean, x_std)
        )
        for k, i in enumerate(idx)
        if np.isfinite(xs[k]).all()
    ]
    raw_generated_feats = [features(*path) for path in generated_paths]
    stopped_paths = [
        pin_dispatch_terminal_stop(
            resample_for_dispatch(
                *reconstruct(
                    xs[k],
                    float(meta_all[i, 0]),
                    x_mean,
                    x_std,
                    force_terminal_stop=bool(stop_flags[k]),
                )
            ),
            bool(stop_flags[k]),
        )
        for k, i in enumerate(idx)
        if np.isfinite(xs[k]).all()
    ]
    shipped_generated_feats = [
        features(*add_default_tremor(*path, args.tremor_amplitude, rng))
        for path in stopped_paths
    ]
    if n_nonfinite:
        print(f"WARNING: {n_nonfinite} non-finite samples dropped\n")

    raw_mean = report("raw flow after dispatch resampling", real_feats, raw_generated_feats)
    shipped_mean = report(
        f"shipped hybrid path (AR(2) amplitude {args.tremor_amplitude:g})",
        real_feats,
        shipped_generated_feats,
    )
    delta = shipped_mean - raw_mean
    print(f"\npost-processing mean-separability delta: {delta:+.1f} points")
    print("0% = indistinguishable on these diagnostics; 100% = complete rank separation")


if __name__ == "__main__":
    main()
