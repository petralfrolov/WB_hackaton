"""Conformal prediction utilities for demand forecasting.

Implements split conformal prediction with per-route, per-horizon calibration:

  All calibration data comes from a single normalised allsteps CSV
  (data/non_conformity_scores_norm_allsteps.csv):
      route_id, horizon, step, score
      where ``score`` = |y_actual - y_hat| / (y_hat + 1)

  load_ncs() extracts the three canonical 2-hour horizons (0-2h, 2-4h, 4-6h)
  from this file.  load_ncs_allsteps() returns the full 12-step data.

  get_margin() denormalises the quantile:
      margin_abs = q_relative * (pred + 1)

  The margin q_alpha is looked up via a three-level chain:
    1. (route_id, horizon)        — route-specific, horizon-specific
    2. ('__global__', horizon)    — cross-route aggregate for this horizon
    3. ('__global__', '__all__')  — full global pool

  The quantile is computed with method='higher' (numpy ≥ 1.22) which
  implements the ceiling-index formula:
      q = scores[ceil((n+1)*alpha) - 1]
  This gives the strict finite-sample coverage guarantee from
  Angelopoulos & Bates (2021) — no interpolation, coverage ≥ alpha.

Reference: https://arxiv.org/abs/2107.07511
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from config import NCS_ALLSTEPS_PATH, CONFORMAL_HORIZONS, ALLSTEP_HORIZON_LABELS

HORIZONS = CONFORMAL_HORIZONS

# All 12 step horizons (step i covers [(i-4)*0.5, (i-4)*0.5+2] hours)
ALLSTEP_HORIZONS = tuple(ALLSTEP_HORIZON_LABELS)

# Type alias: maps (route_id | '__global__', horizon | '__all__') → calibration scores
NcsData = Dict[Tuple[str, str], np.ndarray]


def load_ncs(path: Path = NCS_ALLSTEPS_PATH) -> Tuple[NcsData, bool]:
    """Load 3-horizon non-conformity scores extracted from the allsteps CSV.

    Filters the allsteps data to the three canonical horizons (0-2h, 2-4h, 4-6h).
    Always normalised.  Returns ``(ncs_data, True)``.

    ncs_data contains:
    - ``(route_id, horizon)``       — per-route, per-horizon scores
    - ``('__global__', horizon)``   — aggregated across all routes per horizon
    - ``('__global__', '__all__')`` — full global pool
    """
    is_normalized = True

    df = pd.read_csv(path, comment="#")
    df = df.dropna(subset=["score"]).copy()
    df["route_id"] = df["route_id"].astype(str)
    df["horizon"] = df["horizon"].astype(str)
    df["score"] = df["score"].astype(float)

    # Keep only the three canonical 2h horizons
    df = df[df["horizon"].isin(HORIZONS)]

    ncs: NcsData = {}

    # Per-route, per-horizon
    for (rid, hor), grp in df.groupby(["route_id", "horizon"], sort=False):
        ncs[(rid, hor)] = grp["score"].to_numpy(dtype=float)

    # Global per-horizon aggregate (cross-route)
    for hor in HORIZONS:
        sub = df[df["horizon"] == hor]["score"].to_numpy(dtype=float)
        if len(sub) > 0:
            ncs[("__global__", hor)] = sub

    # Ultimate global fallback
    ncs[("__global__", "__all__")] = df["score"].to_numpy(dtype=float)

    return ncs, is_normalized


def compute_margin(scores: np.ndarray, alpha: float, winsor: float = 0.95) -> float:
    """Conformal prediction margin with strict theoretical coverage guarantee.

    Implements the ceiling-index quantile from split conformal theory:
        level = min(1.0, ceil((n+1)*alpha) / n)
        q     = quantile(scores, level, method='higher')

    Using ``method='higher'`` (numpy ≥ 1.22) avoids linear interpolation,
    returning the actual next observed score.  This ensures marginal coverage
    is *exactly* ≥ alpha for any finite calibration set n.

    ``winsor`` clips scores at that percentile before computing the quantile,
    removing anomalous calibration points (e.g. data quality outliers).
    Set winsor=1.0 to disable winsorizing.
    """
    n = len(scores)
    if n == 0:
        return 0.0
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha == 0.0:
        return 0.0
    if 0.0 < winsor < 1.0:
        cap = float(np.quantile(scores, winsor))
        scores = scores[scores <= cap]
        if len(scores) == 0:
            return 0.0
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * alpha) / n)
    return float(np.quantile(scores, level, method="higher"))


def get_margin(
    ncs: NcsData,
    route_id: str,
    horizon: str,
    alpha: float,
    pred: float = 0.0,
    normalized: bool = False,
    max_margin_factor: float = float("inf"),
) -> float:
    """Resolve the conformal margin for a specific (route, horizon) pair.

    Fallback chain:
      1. (route_id, horizon)        — most specific
      2. ('__global__', horizon)    — horizon-specific aggregate
      3. ('__global__', '__all__')  — global pool

    When ``normalized=True`` the stored scores are relative residuals
    |y-ŷ|/(ŷ+1), so the returned margin is denormalised:
        margin_abs = q_relative * (pred + 1)

    ``max_margin_factor`` caps the final margin at ``factor × pred``
    (e.g. 0.5 → margin ≤ 50% of the point forecast).  Set to inf to disable.
    """
    for key in [
        (route_id, horizon),
        ("__global__", horizon),
        ("__global__", "__all__"),
    ]:
        scores = ncs.get(key)
        if scores is not None and len(scores) > 0:
            q = compute_margin(scores, alpha)
            margin = round(q * (pred + 1)) if normalized else q
            if max_margin_factor < float("inf") and pred > 0:
                margin = min(margin, round(max_margin_factor * pred))
            return margin
    return 0.0


def conformal_interval(pred: float, margin: float) -> Tuple[float, float]:
    """Return (lower, upper) conformal prediction interval.

    lower = max(0, pred − margin),  upper = pred + margin.
    """
    lower = max(0.0, round(pred - margin, 2))
    upper = round(pred + margin, 2)
    return lower, upper


def load_ncs_allsteps(path: Path = NCS_ALLSTEPS_PATH) -> Tuple[NcsData, bool]:
    """Load per-step non-conformity scores from the allsteps CSV.

    CSV format: route_id,horizon,step,score
    Horizons are the 12 step labels (e.g. '-1.5-0.5h', '0-2h', '4-6h').
    Indexed by (route_id, horizon) and ('__global__', horizon).
    Always normalised.
    """
    is_normalized = True

    df = pd.read_csv(path, comment="#")
    df = df.dropna(subset=["score"]).copy()
    df["route_id"] = df["route_id"].astype(str)
    df["horizon"] = df["horizon"].astype(str)
    df["score"] = df["score"].astype(float)

    ncs: NcsData = {}

    for (rid, hor), grp in df.groupby(["route_id", "horizon"], sort=False):
        ncs[(rid, hor)] = grp["score"].to_numpy(dtype=float)

    for hor in ALLSTEP_HORIZONS:
        sub = df[df["horizon"] == hor]["score"].to_numpy(dtype=float)
        if len(sub) > 0:
            ncs[("__global__", hor)] = sub

    ncs[("__global__", "__all__")] = df["score"].to_numpy(dtype=float)
    return ncs, is_normalized
