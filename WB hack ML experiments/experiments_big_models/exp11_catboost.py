"""Experiment 11: CatBoost with MAE loss + extended features."""
from pipeline import run

# CatBoost params
cb_params = {
    "iterations": 3000,
    "learning_rate": 0.05,
    "depth": 8,
    "loss_function": "MAE",
    "eval_metric": "MAE",
    "random_seed": 42,
    "verbose": False,
    "thread_count": -1,
    "od_type": "Iter",
    "od_wait": 100,
    "use_best_model": True,
}

run(
    "exp11_catboost",
    use_catboost=True,
    cb_params=cb_params,
    extended_features=True,
)
