"""Experiment 14: Quantile regression alpha=0.57 — targets zero bias WITHOUT post-hoc correction.

Math reasoning:
- MAE predicts conditional median.  Median < mean for right-skewed target.
- Sum(medians) < sum(targets) → RBias ≈ 0.057 (5.7% under-pred observed).
- Using quantile α ≈ 0.57 (above median) makes sum(predictions) ≈ sum(targets).
- Combined metric: WAPE(0.57) + RBias(0.57) vs WAPE(0.50) + 0 after bias correction.
- Hope: WAPE(0.57) close enough to WAPE(0.50) that total improves.
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

# Don't apply bias correction so we can see the raw RBias achieved by quantile choice
run(
    "exp14_quantile_057",
    lgb_params=params,
    extended_features=True,
    use_bias_correction=True,   # still compute corrected version too
)
