"""Fit a skewed outcome distribution to each player's mean, floor and ceiling.

Each player gets a two-piece normal: one spread below the centre and a different one
above it, cut off at the lowest possible score. Three inputs pin down its three
parameters exactly, so the fit reproduces the projections instead of approximating them.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

_K = 1.0 / np.sqrt(2.0 * np.pi)
_GRID = np.array([NormalDist().inv_cdf((i + 0.5) / 4000) for i in range(4000)])


def fit_players(players: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Return per-player parameters (loc, s_lo, s_hi, lb) plus a fit report."""
    a = -NormalDist().inv_cdf(settings["floor_percentile"])
    b = NormalDist().inv_cdf(settings["ceiling_percentile"])
    denom = 1.0 - _K / a - _K / b
    if a <= 0 or b <= 0 or denom <= 0.05:
        raise ValueError("floor/ceiling percentiles are too close to the middle to fit")

    mean = players["mean"].to_numpy(float)
    floor = players["floor"].to_numpy(float)
    ceil = players["ceiling"].to_numpy(float)
    lb = players["position"].map(settings["lower_bound"]).to_numpy(float)
    lb = np.minimum(lb, np.where(np.isnan(floor), lb, floor))

    # Replace missing or inconsistent floors/ceilings with typical ratios.
    spread = np.maximum(np.abs(mean), 1.0)
    bad = np.isnan(floor) | np.isnan(ceil) | (floor >= mean) | (ceil <= mean)
    floor = np.where(bad, mean - (1 - settings["fallback_floor_ratio"]) * spread, floor)
    ceil = np.where(bad, mean + (settings["fallback_ceiling_ratio"] - 1) * spread, ceil)

    # Solve for the centre so mean, floor and ceiling all hold once scores are cut off
    # at the lower bound. The centre is kept inside the floor-ceiling range.
    width = ceil - floor
    low, high = floor + 0.1 * width, ceil - 0.1 * width
    centre = np.clip((mean - _K * (ceil / b + floor / a)) / denom, low, high)
    grid = _GRID[None, :]
    for _ in range(80):
        s_lo, s_hi = (centre - floor) / a, (ceil - centre) / b
        gap = mean - _sample(centre, s_lo, s_hi, lb, grid).mean(axis=1)
        centre = np.clip(centre + 0.7 * gap / denom, low, high)
    s_lo, s_hi = (centre - floor) / a, (ceil - centre) / b

    # Where the three numbers cannot all hold, keep the mean exact by shifting.
    loc = centre.copy()
    for _ in range(60):
        loc = loc + (mean - _sample(loc, s_lo, s_hi, lb, grid).mean(axis=1))

    fitted_floor = np.maximum(loc - s_lo * a, lb)
    fitted_ceil = np.maximum(loc + s_hi * b, lb)
    tolerance = np.maximum(0.25, 0.03 * width)
    off = (np.abs(fitted_floor - floor) > tolerance) | (np.abs(fitted_ceil - ceil) > tolerance)
    return pd.DataFrame(
        {
            "loc": loc, "s_lo": s_lo, "s_hi": s_hi, "lb": lb,
            "fit_mean": _sample(loc, s_lo, s_hi, lb, _GRID[None, :]).mean(axis=1),
            "fit_floor": fitted_floor, "fit_ceiling": fitted_ceil,
            "used_fallback": bad, "shape_adjusted": off & ~bad,
        },
        index=players.index,
    )


def _sample(loc, s_lo, s_hi, lb, z):
    loc, s_lo, s_hi, lb = (np.asarray(v)[:, None] for v in (loc, s_lo, s_hi, lb))
    return np.maximum(loc + np.where(z < 0, s_lo, s_hi) * z, lb)


def scores_from_normals(fit: pd.DataFrame, z: np.ndarray) -> np.ndarray:
    """Turn standard-normal draws (players x sims) into fantasy scores."""
    return _sample(fit["loc"], fit["s_lo"], fit["s_hi"], fit["lb"], z)
