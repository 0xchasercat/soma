"""
Phase 1 — Dataset extraction and normalization for the conditional flow.

Transforms raw soma capture JSON into a non-degenerate model vector.

Representation
--------------
Each mouse gesture is mapped to a target-relative canonical frame:

  1. translate  : start point -> (0, 0)
  2. rotate     : endpoint direction -> +X axis
  3. resample   : 81 points on a uniform *time* grid (linear interpolation)
  4. difference : 80 spatial deltas per axis
  5. scale      : divide spatial deltas by gesture distance

Velocity information lives entirely in (du, dv) because the time grid is
uniform: a fast segment produces a large spatial delta.

DEGENERACY (why the model vector is 159-d, not 161-d)
-----------------------------------------------------
Three exact constraints are induced by the normalization above:

  * dt is identical for all 80 steps of every gesture (uniform time grid),
    so after dividing by total duration it equals 1/80 *exactly*, for every
    sample in the corpus. Zero variance -> 80 dead dimensions.
  * sum(du) == 1.0 exactly (endpoint sits at +1 after scaling by distance).
  * sum(dv) == 0.0 exactly (endpoint sits on the +X axis after rotation).

A normalizing flow trained on exact negative log-likelihood cannot fit a
density supported on a lower-dimensional manifold: the log-likelihood diverges
and the spline scales collapse to NaN. So the redundant coordinates are
dropped rather than modeled:

    x = [ du except index 40 (79), dv except index 40 (79),
          log(duration_ms) (1) ]  -> 159

One middle delta per axis is dropped and recovered exactly at synthesis time:

    du[40] = 1.0 - sum(du[all indices except 40])
    dv[40] = 0.0 - sum(dv[all indices except 40])

Dropping the final delta made touchdown velocity unmodeled and produced a large
terminal-speed artifact. A middle coordinate satisfies the same mathematical
constraint while keeping the observed first and final deltas in the density.

Total duration becomes a *generated* quantity instead of a Fitts's-law output.
That removes another hand-calibrated component and directly addresses
discriminator feature [7] (actual_duration / fitts_duration).

Outputs
-------
x     : (N, 159) float32, standardized per dimension
c     : (N, 3)   float32, standardized -- [log(distance), log(target_size), terminal_stop]
meta  : (N, 4)   float32, raw          -- [distance_px, target_size_px, duration_ms, terminal_stop]
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

N_STEPS = 80   # spatial deltas per gesture; matches the native ~119 Hz capture
               # rate (8.4ms median interval) with 6.93ms/step at median 665ms
               # duration — 1.21× oversampling, minimal interpolation artifact.
               # N_STEPS=128 was 1.62× oversample and manufactured artificial
               # velocity spikes that corrupted every downstream measurement.
N_FREE = N_STEPS - 1  # free deltas per axis after dropping one constrained value
DROPPED_DELTA_INDEX = N_STEPS // 2
MODEL_DIM = 2 * N_FREE + 1  # 79 du + 79 dv + 1 log-duration = 159
# Canonicalization divides by start-to-end displacement. A zero-displacement
# record has no target axis and cannot enter this target-conditioned reach
# model. It remains valid human evidence for discriminator training; this is a
# representation boundary, not a behavioral quality filter.
MIN_CANONICAL_DISTANCE_PX = 1e-6
MIN_POINTS = 2  # minimum needed for a temporally measurable movement
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


def process_movement(
    mov: dict,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray] | None, str | None]:
    """
    Convert one RawMovement into (x, c, meta).

    Genuine human behavior is never rejected for being short, slow, curved, or
    compound. ``reason`` is populated only when the record cannot be measured
    or represented by a target-directed canonical frame.

    x    : (159,) unstandardized model vector
    c    : (3,)   context -- [log(distance), log(target_size), terminal_stop]
    meta : (4,)   raw -- [distance, target_size, duration_ms, terminal_stop]
    """
    raw = mov.get("trajectory") or []
    points: list[tuple[float, float, float]] = []
    for point in raw:
        if not isinstance(point, dict):
            continue
        try:
            x, y, t = float(point["x"]), float(point["y"]), float(point["tMs"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y) and math.isfinite(t):
            points.append((x, y, t))
    if len(points) < MIN_POINTS:
        return None, "fewer than 2 finite points"

    # Preserve the measured path while repairing timestamp delivery artifacts.
    # At an identical timestamp only the last coordinate is temporally
    # observable, so it replaces the previous coordinate rather than causing
    # the complete human movement to be thrown away.
    points.sort(key=lambda point: point[2])
    distinct: list[tuple[float, float, float]] = []
    for point in points:
        if distinct and point[2] == distinct[-1][2]:
            distinct[-1] = point
        else:
            distinct.append(point)
    if len(distinct) < MIN_POINTS:
        return None, "fewer than 2 distinct timestamps"

    raw_x = np.array([point[0] for point in distinct], dtype=np.float64)
    raw_y = np.array([point[1] for point in distinct], dtype=np.float64)
    raw_t = np.array([point[2] for point in distinct], dtype=np.float64)
    raw_t = raw_t - raw_t[0]
    duration_ms = float(raw_t[-1])
    if duration_ms <= 0:
        return None, "non-positive duration"

    dx = raw_x[-1] - raw_x[0]
    dy = raw_y[-1] - raw_y[0]
    dist = math.hypot(dx, dy)
    if dist <= MIN_CANONICAL_DISTANCE_PX:
        return None, "zero endpoint displacement"

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
    terminal_stop = float(math.hypot(delta_u[-1], delta_v[-1]) <= 1e-6)

    # Drop one constrained middle delta on each axis (recovered at synthesis).
    # The final delta remains explicit so touchdown speed is learned.
    x_vec = np.concatenate(
        [
            np.delete(delta_u, DROPPED_DELTA_INDEX),
            np.delete(delta_v, DROPPED_DELTA_INDEX),
            [math.log(duration_ms)],
        ]
    ).astype(np.float32)

    c_vec = np.array(
        [math.log(dist), math.log(target_size), terminal_stop], dtype=np.float32
    )
    meta = np.array(
        [dist, target_size, duration_ms, terminal_stop], dtype=np.float32
    )

    if not (np.all(np.isfinite(x_vec)) and np.all(np.isfinite(c_vec))):
        return None, "non-finite canonical representation"

    return (x_vec, c_vec, meta), None


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
    group_ids: list[str] = []
    movement_ids: list[str] = []
    unusable: dict[str, int] = {}

    for index, mov in enumerate(movements):
        result, reason = process_movement(mov)
        if result is not None:
            x, c, meta = result
            xs.append(x)
            cs.append(c)
            metas.append(meta)
            group_ids.append(str(mov.get("sessionId") or f"ungrouped-{index}"))
            movement_ids.append(str(mov.get("id") or f"movement-{index}"))
            continue
        key = reason or "unknown structural failure"
        unusable[key] = unusable.get(key, 0) + 1

    if not xs:
        sys.exit("No canonicalizable movements -- check the capture format.")

    x_all = np.stack(xs)  # (N, 159)
    c_all = np.stack(cs)  # (N, 2)
    meta_all = np.stack(metas)  # (N, 3)

    print("\nPreservation summary:")
    print(f"  canonicalized human movements : {len(xs)}")
    for reason, count in sorted(unusable.items()):
        print(f"  unrepresentable ({reason}) : {count}")
    print("  behavioral filters (distance/duration/curvature) : 0")

    # Verify the constraints this parameterization claims to have removed.
    du_full_sum = x_all[:, :N_FREE].sum(axis=1)
    dv_full_sum = x_all[:, N_FREE : 2 * N_FREE].sum(axis=1)
    print("\nConstraint check (middle deltas dropped, so these must NOT be constant):")
    print(f"  sum(du[0..{N_FREE - 1}]) std : {du_full_sum.std():.6f}")
    print(f"  sum(dv[0..{N_FREE - 1}]) std : {dv_full_sum.std():.6f}")

    # Per-dimension standardization. Velocity profile means Δu at t=0 has a
    # systematically different mean than later samples, so per-dimension (not
    # per-channel) statistics are the correct conditioning for this flow.
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

    print(
        f"\nModel vector: {N_FREE} du + {N_FREE} dv + 1 log-duration = "
        f"{x_all.shape[1]} dims"
    )
    print(f"  du  std range : {x_std[:N_FREE].min():.6f} .. {x_std[:N_FREE].max():.6f}")
    print(
        f"  dv  std range : {x_std[N_FREE:2 * N_FREE].min():.6f} .. "
        f"{x_std[N_FREE:2 * N_FREE].max():.6f}"
    )
    print(f"  log-duration  : mean {x_mean[-1]:.4f}  std {x_std[-1]:.4f}")

    print("\nContext [log(distance), log(target_size), terminal_stop]:")
    print(f"  mean: {c_mean}")
    print(f"  std : {c_std}")

    print("\nRaw scalars (min / median / max):")
    for i, name in enumerate(
        ("distance_px", "target_size_px", "duration_ms", "terminal_stop")
    ):
        col = meta_all[:, i]
        print(f"  {name:15s}: {col.min():9.2f} / {np.median(col):9.2f} / {col.max():9.2f}")

    dataset = {
        "x": torch.from_numpy(x_norm.astype(np.float32)),
        "c": torch.from_numpy(c_norm.astype(np.float32)),
        "meta": torch.from_numpy(meta_all),
        "group_ids": group_ids,
        "movement_ids": movement_ids,
        "stats": {
            "x_mean": torch.from_numpy(x_mean.astype(np.float32)),
            "x_std": torch.from_numpy(x_std.astype(np.float32)),
            "c_mean": torch.from_numpy(c_mean.astype(np.float32)),
            "c_std": torch.from_numpy(c_std.astype(np.float32)),
        },
        "layout": {
            "n_steps": N_STEPS,
            "n_free": N_FREE,
            "dropped_delta_index": DROPPED_DELTA_INDEX,
            "model_dim": MODEL_DIM,
            "du": [0, N_FREE],
            "dv": [N_FREE, 2 * N_FREE],
            "log_duration": 2 * N_FREE,
        },
        "source": {
            "capture": Path(args.capture).name,
            "raw_movements": len(movements),
            "canonicalized_movements": len(xs),
            "unrepresentable": unusable,
            "behavioral_filters": [],
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
