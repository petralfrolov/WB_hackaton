"""Conformal prediction utilities for demand forecasting.

Implements split conformal prediction:
  - Calibration phase: compute non-conformity scores = |y_actual - y_pred|
    on a held-out calibration set and save to data/non_conformity_scores.csv
  - Inference phase: for confidence level alpha, the prediction interval is
      [max(0, pred - q_alpha), pred + q_alpha]
    where q_alpha = ceil((n+1)*alpha)/n -th empirical quantile of calibration scores.

Reference: Venn-Abers / Split Conformal, Angelopoulos & Bates (2021).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

NCS_DEFAULT_PATH = Path("data/non_conformity_scores.csv")

_DEFAULT_FALLBACK_SCORES = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)


def load_ncs(path: Path = NCS_DEFAULT_PATH) -> np.ndarray:
    """Load non-conformity scores from CSV.

    The CSV must have a ``score`` column (absolute residuals on calibration set).
    Falls back to a fixed default array if the file is missing or malformed.
    """
    try:
        df = pd.read_csv(path, comment="#")
        if "score" not in df.columns:
            return _DEFAULT_FALLBACK_SCORES.copy()
        scores = df["score"].dropna().to_numpy(dtype=float)
        return scores if len(scores) > 0 else _DEFAULT_FALLBACK_SCORES.copy()
    except Exception:
        return _DEFAULT_FALLBACK_SCORES.copy()


def compute_margin(scores: np.ndarray, alpha: float) -> float:
    """Compute conformal prediction margin for coverage level *alpha*.

    Uses the split-conformal formula:
        q = quantile(scores, ceil((n+1)*alpha) / n)
    clamped to [0, inf).

    Parameters
    ----------
    scores : ndarray
        Non-conformity scores from calibration (absolute residuals).
    alpha : float
        Desired marginal coverage, e.g. 0.90 for 90 %.

    Returns
    -------
    float
        Margin q such that coverage is approximately >= alpha.
    """
    n = len(scores)
    if n == 0:
        return 0.0
    alpha = float(np.clip(alpha, 0.0, 1.0))
    # ceiling quantile for finite-sample guarantee
    level = min(1.0, np.ceil((n + 1) * alpha) / n)
    return float(np.quantile(scores, level))


def conformal_interval(pred: float, margin: float) -> Tuple[float, float]:
    """Return (lower, upper) conformal prediction interval.

    Parameters
    ----------
    pred : float
        Point forecast.
    margin : float
        Conformal margin q_alpha.

    Returns
    -------
    tuple[float, float]
        (lower, upper) where lower = max(0, pred - margin).
    """
    lower = max(0.0, round(pred - margin, 2))
    upper = round(pred + margin, 2)
    return lower, upper
