"""Model training, prediction, and submission generation."""

import numpy as np
import pandas as pd
import lightgbm as lgb
from config import FUTURE_TARGET_COLS, LGB_PARAMS

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except (ImportError, ValueError):
    CATBOOST_AVAILABLE = False


def train_lgb_models(X_fit, y_fit, X_valid, y_valid, lgb_params=None):
    """Train one LGBMRegressor per forecast step with early stopping."""
    params = lgb_params or LGB_PARAMS
    models = {}
    for step_col in FUTURE_TARGET_COLS:
        m = lgb.LGBMRegressor(**params)
        m.fit(
            X_fit, y_fit[step_col],
            eval_set=[(X_valid, y_valid[step_col])],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0),  # silent
            ],
        )
        models[step_col] = m
        val_scores = m.best_score_.get("valid_0", {})
        score_val = next(iter(val_scores.values()), float("nan"))
        print(f"  {step_col:20s}  best_iter={m.best_iteration_:4d}  "
              f"best_score={score_val:.4f}")
    return models


def train_catboost_models(X_fit, y_fit, X_valid, y_valid, cat_features, cb_params=None):
    """Train one CatBoostRegressor per forecast step with early stopping."""
    if not CATBOOST_AVAILABLE:
        raise ImportError("catboost not installed")

    # CatBoost needs integer indices for categorical features
    cat_feat_indices = [list(X_fit.columns).index(c) for c in cat_features if c in X_fit.columns]
    # Convert categoricals to string for CatBoost
    X_fit_cb = X_fit.copy()
    X_valid_cb = X_valid.copy()
    for col in cat_features:
        if col in X_fit_cb.columns:
            X_fit_cb[col] = X_fit_cb[col].astype(str)
            X_valid_cb[col] = X_valid_cb[col].astype(str)

    default_params = {
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
    if cb_params:
        default_params.update(cb_params)

    models = {}
    for step_col in FUTURE_TARGET_COLS:
        train_pool = cb.Pool(X_fit_cb, y_fit[step_col].clip(lower=0),
                             cat_features=cat_feat_indices)
        valid_pool = cb.Pool(X_valid_cb, y_valid[step_col].clip(lower=0),
                             cat_features=cat_feat_indices)
        m = cb.CatBoostRegressor(**default_params)
        m.fit(train_pool, eval_set=valid_pool)
        models[step_col] = m
        best_score = m.best_score_["validation"]["MAE"]
        print(f"  {step_col:20s}  best_iter={m.best_iteration_:4d}  "
              f"best_score={best_score:.4f}")
    return models


def train_dirrec_models(X_fit, y_fit, X_valid, y_valid, lgb_params=None):
    """
    DirRec training: each step model gets previous-step predictions as an extra feature.
    Uses in-sample step-1..k-1 predictions for X_fit (slight bias, but adds signal).
    Uses model predictions sequentially for X_valid and X_test.
    """
    params = lgb_params or LGB_PARAMS
    models = {}
    X_fit_aug = X_fit.copy()
    X_valid_aug = X_valid.copy()

    for i, step_col in enumerate(FUTURE_TARGET_COLS):
        m = lgb.LGBMRegressor(**params)
        m.fit(
            X_fit_aug, y_fit[step_col],
            eval_set=[(X_valid_aug, y_valid[step_col])],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        models[step_col] = m
        val_scores = m.best_score_.get("valid_0", {})
        score_val = next(iter(val_scores.values()), float("nan"))
        print(f"  {step_col:20s}  best_iter={m.best_iteration_:4d}  "
              f"best_score={score_val:.4f}  features={X_fit_aug.shape[1]}")

        # Add this step's prediction as feature for the next step
        if i < len(FUTURE_TARGET_COLS) - 1:
            prev_col = f"prev_{step_col}"
            fit_preds = np.clip(m.predict(X_fit_aug), 0, None)
            val_preds = np.clip(m.predict(X_valid_aug), 0, None)
            X_fit_aug = X_fit_aug.copy()
            X_valid_aug = X_valid_aug.copy()
            X_fit_aug[prev_col] = fit_preds
            X_valid_aug[prev_col] = val_preds

    # Store the augmented X_valid for evaluation
    models["_X_valid_aug"] = X_valid_aug
    return models


def predict_steps_dirrec(models, X: pd.DataFrame) -> pd.DataFrame:
    """Sequential prediction using DirRec models — adds previous step pred as feature."""
    X_aug = X.copy()
    preds = {}
    for i, step_col in enumerate(FUTURE_TARGET_COLS):
        p = np.clip(models[step_col].predict(X_aug), 0, None)
        preds[step_col] = p
        if i < len(FUTURE_TARGET_COLS) - 1:
            prev_col = f"prev_{step_col}"
            X_aug = X_aug.copy()
            X_aug[prev_col] = p
    return pd.DataFrame(preds, index=X.index)


def predict_steps_catboost(models, X: pd.DataFrame, cat_features: list) -> pd.DataFrame:
    """Predict using CatBoost models — convert categoricals to str first."""
    X_cb = X.copy()
    for col in cat_features:
        if col in X_cb.columns:
            X_cb[col] = X_cb[col].astype(str)
    preds = {col: np.clip(models[col].predict(X_cb), 0, None) for col in FUTURE_TARGET_COLS}
    return pd.DataFrame(preds, index=X.index)


def predict_steps(models, X: pd.DataFrame) -> pd.DataFrame:
    """Predict all forecast steps, clip negatives to 0."""
    preds = {col: np.clip(models[col].predict(X), 0, None) for col in FUTURE_TARGET_COLS}
    return pd.DataFrame(preds, index=X.index)


def compute_bias_factor(y_valid: pd.DataFrame, valid_pred_df: pd.DataFrame) -> float:
    """Compute multiplicative bias correction factor from validation."""
    yt = y_valid.to_numpy().flatten()
    yp = valid_pred_df.to_numpy().flatten()
    factor = yt.sum() / (yp.sum() + 1e-8)
    return factor


def build_submission(test_pred_df, X_test, inference_ts, test_df, bias_factor=1.0):
    """Convert step-wise predictions to competition submission format."""
    corrected = (test_pred_df * bias_factor).clip(lower=0).copy()
    # Cast route_id back to original dtype (may be categorical after LGB encoding)
    rid = X_test["route_id"].values
    if hasattr(rid, "astype"):
        rid = rid.astype(test_df["route_id"].dtype)
    corrected["route_id"] = rid

    forecast_df = corrected.melt(
        id_vars="route_id",
        value_vars=[c for c in corrected.columns if c.startswith("target_step_")],
        var_name="step",
        value_name="forecast",
    )
    forecast_df["step_num"] = forecast_df["step"].str.extract(r"(\d+)").astype(int)
    forecast_df["timestamp"] = inference_ts + pd.to_timedelta(forecast_df["step_num"] * 30, unit="m")
    forecast_df = (
        forecast_df[["route_id", "timestamp", "forecast"]]
        .sort_values(["route_id", "timestamp"])
        .reset_index(drop=True)
    )
    result = test_df.merge(forecast_df, "outer")[["id", "forecast"]]
    result = result.rename(columns={"forecast": "y_pred"})
    assert result["id"].isna().sum() == 0, "Missing IDs in submission!"
    return result
