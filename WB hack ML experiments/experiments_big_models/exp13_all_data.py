"""Experiment 13: Use ALL 90 days of training data + extended features + MAE.

Hypothesis: previous attempt with more data (exp03, 60 days) was worse because
weekly lags weren't there. NOW with lag_336/lag_672, extra history provides
fully-populated weekly lags for ALL training rows, giving the model 90 days
of properly-featured samples. MAX_TRAIN_ROWS=2M handles memory.
"""
import sys
sys.path.insert(0, ".")

# Temporarily patch config to use all data
import config
config.TRAIN_DAYS = 90

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
    "exp13_all_data",
    lgb_params=params,
    extended_features=True,
)
