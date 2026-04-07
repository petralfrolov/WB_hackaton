"""Experiment 16: DirRec + 6-pair deconvolution + extended features + quantile 0.57.

DirRec: each step_k model receives the previous step's prediction as an extra feature.
This allows the model to "chain" predictions: step_2 knows what step_1 predicted,
giving direct continuity information unavailable in the Direct approach.
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
    "exp16_dirrec",
    lgb_params=params,
    extended_features=True,
    use_dirrec=True,
)
