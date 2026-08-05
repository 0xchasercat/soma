"""
Phase 1 — Dataset extraction and normalization for the conditional flow.

Transforms raw soma capture JSON into a non-degenerate model vector.

Representation
--------------
Each mouse gesture is mapped to a target-relative canonical frame:

  1. translate  : start point -> (0, 0)
  2. rotate     : endpoint direction -> +X axis
  3. resample   : 129 points on a uniform *time* grid (linear interpolation)
  4. difference : 128 spatial deltas per axis
  5. scale      : divide spatial deltas by gesture distance

Velocity information lives entirely in (du, dv) because the time grid is
uniform: a fast segment produces a large spatial delta.

DEGENERACY (why the model vector is 255-d, not 384-d)
-----------------------------------------------------
Three exact constraints are induced by the normalization above:

  * dt is identical for all 128 steps of every gesture (uniform time grid),
    so after dividing by total duration it equals 1/128 *exactly*, for every
    sample in the corpus. Zero variance -> 128 dead dimensions.
  * sum(du) == 1.0 exactly (endpoint sits at +1 after scaling by distance).
  * sum(dv) == 0.0 exactly (endpoint sits on the +X axis after rotation).

A normalizing flow trained on exact negative log-likelihood cannot fit a
density supported on a lower-dimensional manifold: the log-likelihood diverges
and the spline scales collapse to NaN. So the redundant coordinates are
dropped rather than modeled:

    x = [ du[0..126] (127), dv[0..126] (127), log(duration_ms) (1) ]  -> 255

The dropped final deltas are recovered exactly at synthesis time:

    du[127] = 1.0 - sum(du[0..126])
    dv[127] = 0.0 - sum(dv[0..126])

Total duration becomes a *generated* quantity instead of a Fitts's-law output.
That removes another hand-calibrated component and directly addresses
discriminator feature [7] (actual_duration / fitts_duration).

Outputs
-------
x     : (N, 255) float32, standardized per dimension
c     : (N, 2)   float32, standardized -- [log(distance), log(target_size)]
meta  : (N, 3)   float32, raw          -- [distance_px, target_size_px, duration_ms]
stats : mean/std tensors required to invert the standardization

`meta` is unstandardized on purpose: the Phase 3 audit needs real gestures in
pixel space to extract overshoot / peak-timing / bell-profile features.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

# ── Constants ─────────────────────────────────────────────────────────────────

N_STEPS = 128  # spatial deltas per gesture (-> 129 resampled sample points)
N_FREE = N_STEPS - 1  # free deltas per axis after dropping the constrained tail
MODEL_DIM = 2 * N_FREE + 1  # 127 du + 127 dv + 1 log-duration = 255
MIN_DIST_PX = 20.0  # reject sub-threshold gestures (hover jitter, not aimed movement)
MAX_DURATION_MS = 5000.0  # reject parked-pointer gestures (median 11 px/s past 10s)
MIN_POINTS = 4  # reject trajectories too short to interpolate meaningfully
DEFAULT_TARGET_SIZE_PX = 24.0  # fallback when the capture bounding box is zeroed
STD_FLOOR = 1e-6  # guard against divide-by-zero on near-constant dimensions

# ── Helpers ───────────────────────────────────────────────────────────────────


def resample_uniform(
    xs: np.ndarray, ys: np.ndarray, ts: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Resample (x, y) at n+1 uniformly spaced time points via linear interpolation.

    Returns (xu, yu, total_duration). `ts` must be monotonically non-decreasing
    and start at 0.
    """
    total = float(ts[-1])
    t_uniform = np.linspace(0.0, total, n + 1)
    xu = np.interp(t_uniform, ts, xs)
    yu = np.interp(t_uniform, ts, ys)
    return xu, yu, total


def process_movement(mov: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Convert one RawMovement into (x, c, meta), or None when filtered out.

    x    : (255,) unstandardized model vector
    c    : (2,)   unstandardized context -- [log(distance), log(target_size)]
    meta : (3,)   raw scalars            -- [distance, target_size, duration_ms]
    """
    traj = mov.get("trajectory") or []
    if len(traj) < MIN_POINTS:
        return None

    raw_x = np.array([p["x"] for p in traj], dtype=np.float64)
    raw_y = np.array([p["y"] for p in traj], dtype=np.float64)
    raw_t = np.array([p["tMs"] for p in traj], dtype=np.float64)

    # Timestamps relative to gesture start; np.interp requires sorted x-coords.
    raw_t = raw_t - raw_t[0]
    if not np.all(np.diff(raw_t) >= 0):
        order = np.argsort(raw_t, kind="stable")
        raw_t, raw_x, raw_y = raw_t[order], raw_x[order], raw_y[order]
    duration_ms = float(raw_t[-1])
    if duration_ms <= 0 or duration_ms > MAX_DURATION_MS:
        return None

    dx = raw_x[-1] - raw_x[0]
    dy = raw_y[-1] - raw_y[0]
    dist = math.hypot(dx, dy)
    if dist < MIN_DIST_PX:
        return None

    # Target size: min(width, height) is the Fitts-relevant extent along the
    # approach axis. 8.9% of captured gestures carry a zeroed bounding box.
    bb = (mov.get("clickTarget") or {}).get("boundingBox") or {}
    w, h = bb.get("width", 0) or 0, bb.get("height", 0) or 0
    target_size = min(w, h) if (w > 0 and h > 0) else DEFAULT_TARGET_SIZE_PX
    target_size = max(float(target_size), 1.0)

    # Canonical frame: translate start to origin, rotate endpoint onto +X.
    tx = raw_x - raw_x[0]
    ty = raw_y - raw_y[0]
    theta = math.atan2(dy, dx)
    cos_t, sin_t = math.cos(-theta), math.sin(-theta)
    ux = cos_t * tx - sin_t * ty
    vy = sin_t * tx + cos_t * ty

    xu, yu, total = resample_uniform(ux, vy, raw_t, N_STEPS)

    # Spatial deltas, scaled to unit distance. sum(du) == 1, sum(dv) == 0.
    delta_u = np.diff(xu) / dist
    delta_v = np.diff(yu) / dist

    # Drop the constrained tail delta on each axis (recovered at synthesis).
    x_vec = np.concatenate(
        [delta_u[:N_FREE], delta_v[:N_FREE], [math.log(duration_ms)]]
    ).astype(np.float32)

    c_vec = np.array([math.log(dist), math.log(target_size)], dtype=np.float32)
    meta = np.array([dist, target_size, duration_ms], dtype=np.float32)

    if not (np.all(np.isfinite(x_vec)) and np.all(np.isfinite(c_vec))):
        return None

    return x_vec, c_vec, meta


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare conditional-flow training data from a soma capture JSON"
    )
    parser.add_argument("capture", help="Path to soma capture JSON")
    parser.add_argument("--out", default="flow/flow_dataset.pt", help="Output .pt path")
    args = parser.parse_args()

    print(f"Loading {args.capture} ...")
    with open(args.capture) as f:
        data = json.load(f)

    movements = data.get("movements", [])
    print(f"Raw movements: {len(movements)}")

    xs: list[np.ndarray] = []
    cs: list[np.ndarray] = []
    metas: list[np.ndarray] = []
    n_short_traj = n_short_dist = n_bad_time = n_long_dur = 0

    for mov in movements:
        result = process_movement(mov)
        if result is not None:
            x, c, meta = result
            xs.append(x)
            cs.append(c)
            metas.append(meta)
            continue

        # Attribute the rejection for the summary table.
        traj = mov.get("trajectory") or []
        if len(traj) < MIN_POINTS:
            n_short_traj += 1
        elif math.hypot(
            traj[-1]["x"] - traj[0]["x"], traj[-1]["y"] - traj[0]["y"]
        ) < MIN_DIST_PX:
            n_short_dist += 1
        elif traj[-1]["tMs"] - traj[0]["tMs"] > MAX_DURATION_MS:
            n_long_dur += 1
        else:
            n_bad_time += 1

    if not xs:
        sys.exit("No usable gestures after filtering -- check the capture format.")

    x_all = np.stack(xs)  # (N, 255)
    c_all = np.stack(cs)  # (N, 2)
    meta_all = np.stack(metas)  # (N, 3)

    print("\nFilter summary:")
    print(f"  trajectory < {MIN_POINTS} points : {n_short_traj}")
    print(f"  distance < {MIN_DIST_PX:g}px        : {n_short_dist}")
    print(f"  duration > {MAX_DURATION_MS:g}ms       : {n_long_dur}")
    print(f"  non-positive duration / nan  : {n_bad_time}")
    print(f"  kept                         : {len(xs)}")

    # Verify the constraints this parameterization claims to have removed.
    du_full_sum = x_all[:, :N_FREE].sum(axis=1)
    dv_full_sum = x_all[:, N_FREE : 2 * N_FREE].sum(axis=1)
    print("\nConstraint check (tail deltas dropped, so these must NOT be constant):")
    print(f"  sum(du[0..126]) std : {du_full_sum.std():.6f}")
    print(f"  sum(dv[0..126]) std : {dv_full_sum.std():.6f}")

    # Per-dimension standardization. Velocity profile means Δu at t=0 has a
    # systematically different mean than at t=64, so per-dimension (not
    # per-channel) statistics are the correct conditioning for a 255-d flow.
    x_mean = x_all.mean(axis=0)
    x_std = x_all.std(axis=0)
    n_floored = int((x_std < STD_FLOOR).sum())
    if n_floored:
        print(f"  WARNING: {n_floored} dimension(s) below std floor {STD_FLOOR:g}")
    x_std = np.where(x_std < STD_FLOOR, 1.0, x_std)
    x_norm = (x_all - x_mean) / x_std

    c_mean = c_all.mean(axis=0)
    c_std = np.where(c_all.std(axis=0) < STD_FLOOR, 1.0, c_all.std(axis=0))
    c_norm = (c_all - c_mean) / c_std

    print("\nModel vector: 127 du + 127 dv + 1 log-duration = " f"{x_all.shape[1]} dims")
    print(f"  du  std range : {x_std[:N_FREE].min():.6f} .. {x_std[:N_FREE].max():.6f}")
    print(
        f"  dv  std range : {x_std[N_FREE:2 * N_FREE].min():.6f} .. "
        f"{x_std[N_FREE:2 * N_FREE].max():.6f}"
    )
    print(f"  log-duration  : mean {x_mean[-1]:.4f}  std {x_std[-1]:.4f}")

    print("\nContext [log(distance), log(target_size)]:")
    print(f"  mean: {c_mean}")
    print(f"  std : {c_std}")

    print("\nRaw scalars (min / median / max):")
    for i, name in enumerate(("distance_px", "target_size_px", "duration_ms")):
        col = meta_all[:, i]
        print(f"  {name:15s}: {col.min():9.2f} / {np.median(col):9.2f} / {col.max():9.2f}")

    dataset = {
        "x": torch.from_numpy(x_norm.astype(np.float32)),
        "c": torch.from_numpy(c_norm.astype(np.float32)),
        "meta": torch.from_numpy(meta_all),
        "stats": {
            "x_mean": torch.from_numpy(x_mean.astype(np.float32)),
            "x_std": torch.from_numpy(x_std.astype(np.float32)),
            "c_mean": torch.from_numpy(c_mean.astype(np.float32)),
            "c_std": torch.from_numpy(c_std.astype(np.float32)),
        },
        "layout": {
            "n_steps": N_STEPS,
            "n_free": N_FREE,
            "model_dim": MODEL_DIM,
            "du": [0, N_FREE],
            "dv": [N_FREE, 2 * N_FREE],
            "log_duration": 2 * N_FREE,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, out_path)
    print(f"\nSaved: {out_path}")
    print(f"  x    {tuple(dataset['x'].shape)}")
    print(f"  c    {tuple(dataset['c'].shape)}")
    print(f"  meta {tuple(dataset['meta'].shape)}")


if __name__ == "__main__":
    main()
