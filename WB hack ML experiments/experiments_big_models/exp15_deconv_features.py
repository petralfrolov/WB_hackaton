"""Experiment 15: Deconvolved 30-min shipment features + quantile 0.57.

Key insight: target_2h is a 4-period rolling sum. By algebraically deconvolving it,
we recover individual 30-min shipment estimates s(t), s(t-1), s(t-2), s(t-3).
For step_1 prediction: target_step_1 = s(t-2) + s(t-1) + s(t) + s(t+1)
                                      = target_lag_0 - s(t-3) + s(t+1)
The model can now reason directly about what enters/leaves the 2h window.
"""
from pipeline import run

params = {
    "n_estimators": 5000,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "quantile",
    "alpha": 0.57,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}

run(
    "exp15_deconv_features",
    lgb_params=params,
    extended_features=True,
)
