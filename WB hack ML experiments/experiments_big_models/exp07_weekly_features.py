"""Experiment 07: Extended weekly lags + route-level target encoding + MAE objective."""
from pipeline import run

params = {
    "n_estimators": 3000,
    "learning_rate": 0.05,
    "num_leaves": 127,
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
    "exp07_weekly_features",
    lgb_params=params,
    extended_features=True,
)
