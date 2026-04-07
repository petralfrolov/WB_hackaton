"""Experiment 08: More trees (5000) + more leaves (255) — step_1 hit the iter limit in exp07."""
from pipeline import run

params = {
    "n_estimators": 5000,
    "learning_rate": 0.05,
    "num_leaves": 255,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "mae",
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}

run(
    "exp08_more_capacity",
    lgb_params=params,
    extended_features=True,
)
