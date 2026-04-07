"""Metric calculation."""

import numpy as np


class WapePlusRbias:
    """WAPE + |Relative Bias| — competition metric."""

    @property
    def name(self) -> str:
        return "wape_plus_rbias"

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        sum_true = y_true.sum()
        if sum_true == 0:
            return float("inf")
        wape = np.abs(y_pred - y_true).sum() / sum_true
        rbias = abs(y_pred.sum() / sum_true - 1)
        return wape + rbias

    def calculate_components(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Return (wape, rbias, total)."""
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        sum_true = y_true.sum()
        if sum_true == 0:
            return float("inf"), float("inf"), float("inf")
        wape = np.abs(y_pred - y_true).sum() / sum_true
        rbias = abs(y_pred.sum() / sum_true - 1)
        return wape, rbias, wape + rbias
