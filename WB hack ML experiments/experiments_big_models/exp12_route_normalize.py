"""Experiment 12: Route-normalized target transform + MAE + extended features.

Hypothesis: dividing each route's target by its historical mean makes the model
learn a universal relative pattern across routes. The model sees 1.0 as 'normal',
0.5 as 'quiet', 2.0 as 'busy'. This should reduce between-route variance and
let gradient boosting generalize better.
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
    "objective": "mae",
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}

run(
    "exp12_route_normalize",
    lgb_params=params,
    extended_features=True,
    route_normalize=True,
)
