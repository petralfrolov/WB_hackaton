"""Data loading, preprocessing, and train/valid/test splitting."""

import pandas as pd
import numpy as np
from config import (
    CONFIG, TARGET_COL, FORECAST_POINTS, FUTURE_TARGET_COLS,
    TRAIN_DAYS, MAX_TRAIN_ROWS, VALID_FRAC, RANDOM_STATE,
)


def load_data():
    """Load and sort train/test parquet files."""
    train_df = pd.read_parquet(CONFIG["train_path"])
    test_df = pd.read_parquet(CONFIG["test_path"])

    train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])

    train_df = train_df.sort_values(["route_id", "timestamp"]).reset_index(drop=True)
    test_df = test_df.sort_values(["route_id", "timestamp"]).reset_index(drop=True)

    print(f"Train shape: {train_df.shape}  |  Test shape: {test_df.shape}")
    print(f"Train dates: {train_df['timestamp'].min()} -> {train_df['timestamp'].max()}")
    print(f"Test  dates: {test_df['timestamp'].min()} -> {test_df['timestamp'].max()}")
    return train_df, test_df


def create_future_targets(train_df: pd.DataFrame) -> pd.DataFrame:
    """Shift target to create multi-step supervision columns."""
    g = train_df.groupby("route_id", sort=False)
    for step in range(1, FORECAST_POINTS + 1):
        train_df[f"target_step_{step}"] = g[TARGET_COL].shift(-step)
    return train_df


def build_feature_cols(train_df: pd.DataFrame):
    """Return list of feature column names (everything except target/timestamp/id/future)."""
    exclude = {TARGET_COL, "timestamp", "id", *FUTURE_TARGET_COLS}
    return [c for c in train_df.columns if c not in exclude]


def split_data(train_df: pd.DataFrame, feature_cols: list, train_days: int = None):
    """
    Create fit / valid / test-inference matrices.
    Returns (X_fit, y_fit, X_valid, y_valid, X_test, inference_ts, test_model_df)

    Args:
        train_days: override global TRAIN_DAYS constant (e.g. for multi-window experiments).
    """
    _train_days = train_days if train_days is not None else TRAIN_DAYS

    supervised_df = train_df.dropna(subset=FUTURE_TARGET_COLS).copy()
    print(f"Supervised rows (non-NaN future targets): {len(supervised_df)}")

    train_model_df = supervised_df[feature_cols + ["timestamp"] + FUTURE_TARGET_COLS].copy()
    train_model_df = train_model_df.rename(columns={"timestamp": "source_timestamp"})

    ts_max = train_model_df["source_timestamp"].max()
    ts_start = ts_max - pd.Timedelta(days=_train_days)
    train_model_df = train_model_df[train_model_df["source_timestamp"] >= ts_start].copy()
    train_model_df = train_model_df.sort_values("source_timestamp")

    split_point = train_model_df["source_timestamp"].quantile(VALID_FRAC)
    fit_df = train_model_df[train_model_df["source_timestamp"] <= split_point].copy()
    valid_df = train_model_df[train_model_df["source_timestamp"] > split_point].copy()

    if len(fit_df) > MAX_TRAIN_ROWS:
        fit_df = fit_df.sample(MAX_TRAIN_ROWS, random_state=RANDOM_STATE)

    print(f"Fit rows: {len(fit_df)}  |  Valid rows: {len(valid_df)}")

    X_fit = fit_df[feature_cols].copy()
    y_fit = fit_df[FUTURE_TARGET_COLS].copy()
    X_valid = valid_df[feature_cols].copy()
    y_valid = valid_df[FUTURE_TARGET_COLS].copy()

    # Test = last timestamp features for inference
    inference_ts = train_df["timestamp"].max()
    test_model_df = train_df[train_df["timestamp"] == inference_ts]
    X_test = test_model_df[feature_cols].copy()
    print(f"Inference timestamp: {inference_ts}  |  Test rows: {len(X_test)}")

    return X_fit, y_fit, X_valid, y_valid, X_test, inference_ts, test_model_df


def encode_categoricals(X_fit, X_valid, X_test, feature_cols):
    """Convert *_id columns to pd.Categorical for LightGBM native handling."""
    cat_features = [c for c in feature_cols if c.endswith("_id")]
    for col in cat_features:
        all_cats = pd.concat([X_fit[col], X_valid[col], X_test[col]]).astype(str).unique()
        cat_dtype = pd.CategoricalDtype(categories=all_cats)
        X_fit[col] = X_fit[col].astype(str).astype(cat_dtype)
        X_valid[col] = X_valid[col].astype(str).astype(cat_dtype)
        X_test[col] = X_test[col].astype(str).astype(cat_dtype)
    return X_fit, X_valid, X_test, cat_features
