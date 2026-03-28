"""Standalone inference pipeline for exp_best_features_full.

Produces three forecasts for a selected route_id and timestamp using the saved
ensemble of LightGBM step models:
1) 0-2h ahead  -> target_step_1
2) 2-4h ahead  -> target_step_5
3) 4-6h ahead  -> target_step_9

This file is fully independent from project modules (no local imports).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import lightgbm as lgb  # noqa: F401 - needed for unpickling LGB models
import numpy as np
import pandas as pd


# -----------------------------
# Hardcoded config (team track)
# -----------------------------
TRACK = "team"
TARGET_COL = "target_2h"
FORECAST_POINTS = 10
FUTURE_TARGET_COLS = [f"target_step_{i}" for i in range(1, FORECAST_POINTS + 1)]

DEFAULT_TRAIN_PATH = "train_team_track.parquet"
DEFAULT_MODELS_DIR = "models/exp_best_features_full"


def make_features(df: pd.DataFrame, extended: bool = True) -> pd.DataFrame:
    """Replicate training feature engineering used in experiments."""
    df = df.sort_values(["route_id", "timestamp"]).copy()

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.int8)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["halfhour_slot"] = df["hour"] * 2 + df["timestamp"].dt.minute // 30

    g = df.groupby("route_id", sort=False)

    # Target lags
    base_lags = [1, 2, 4, 8, 16, 48]
    if extended:
        base_lags += [96, 192, 336]
    for lag in base_lags:
        df[f"target_lag_{lag}"] = g[TARGET_COL].shift(lag)

    # Rolling stats
    for w in [4, 8, 16, 48]:
        df[f"target_roll_mean_{w}"] = g[TARGET_COL].transform(
            lambda x, w=w: x.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"target_roll_std_{w}"] = g[TARGET_COL].transform(
            lambda x, w=w: x.shift(1).rolling(w, min_periods=2).std().fillna(0)
        )

    if extended:
        for w in [96, 336]:
            df[f"target_roll_mean_{w}"] = g[TARGET_COL].transform(
                lambda x, w=w: x.shift(1).rolling(w, min_periods=1).mean()
            )

    # EMA
    df["target_ema_8"] = g[TARGET_COL].transform(
        lambda x: x.shift(1).ewm(span=8, adjust=False).mean()
    )
    df["target_ema_24"] = g[TARGET_COL].transform(
        lambda x: x.shift(1).ewm(span=24, adjust=False).mean()
    )
    if extended:
        df["target_ema_96"] = g[TARGET_COL].transform(
            lambda x: x.shift(1).ewm(span=96, adjust=False).mean()
        )

    # Diff
    df["target_diff_1"] = g[TARGET_COL].diff(1)
    df["target_diff_2"] = g[TARGET_COL].diff(2)
    df["target_diff_4"] = g[TARGET_COL].diff(4)

    # Rolling min/max
    for w in [8, 48]:
        df[f"target_roll_min_{w}"] = g[TARGET_COL].transform(
            lambda x, w=w: x.shift(1).rolling(w, min_periods=1).min()
        )
        df[f"target_roll_max_{w}"] = g[TARGET_COL].transform(
            lambda x, w=w: x.shift(1).rolling(w, min_periods=1).max()
        )

    # Status features
    status_cols = sorted(
        [
            c
            for c in df.columns
            if c.startswith("status_")
            and not c.endswith("_ratio")
            and "_lag" not in c
            and "_roll" not in c
        ]
    )
    if status_cols:
        df["status_sum"] = df[status_cols].sum(axis=1)
        for col in status_cols:
            df[f"{col}_ratio"] = df[col] / (df["status_sum"] + 1e-6)
            df[f"{col}_lag1"] = g[col].shift(1)
            df[f"{col}_lag2"] = g[col].shift(2)
            df[f"{col}_roll_mean_8"] = g[col].transform(
                lambda x: x.shift(1).rolling(8, min_periods=1).mean()
            )
        if len(status_cols) >= 2:
            df["status_first"] = df[status_cols[0]]
            df["status_last"] = df[status_cols[-1]]
            df["status_last_ratio"] = df["status_last"] / (df["status_sum"] + 1e-6)

    # Deconvolution features
    n_deconv_pairs = 12
    for offset, name in [(0, "s_t0"), (1, "s_t1"), (2, "s_t2"), (3, "s_t3")]:
        terms = []
        for k in range(n_deconv_pairs):
            lag_add = offset + 4 * k
            lag_sub = offset + 4 * k + 1
            lag_add_name = f"target_lag_{lag_add}"
            lag_sub_name = f"target_lag_{lag_sub}"
            if lag_add_name not in df.columns:
                df[lag_add_name] = g[TARGET_COL].shift(lag_add)
            if lag_sub_name not in df.columns:
                df[lag_sub_name] = g[TARGET_COL].shift(lag_sub)
            terms.append(df[lag_add_name])
            terms.append(-df[lag_sub_name])

        s = terms[0]
        for t in terms[1:]:
            s = s + t
        df[f"deconv_{name}"] = s.clip(lower=0)

    # Office-level agg
    office_agg = (
        df.groupby(["office_from_id", "timestamp"])[TARGET_COL]
        .agg(office_target_mean="mean", office_target_sum="sum", office_target_std="std")
        .reset_index()
    )
    df = df.merge(office_agg, on=["office_from_id", "timestamp"], how="left")
    df["office_target_std"] = df["office_target_std"].fillna(0)

    if extended:
        # Route-level target encoding
        route_stats = (
            df.groupby("route_id")[TARGET_COL]
            .agg(
                route_target_mean="mean",
                route_target_median="median",
                route_target_std="std",
                route_target_p75=lambda x: x.quantile(0.75),
                route_target_p25=lambda x: x.quantile(0.25),
            )
            .reset_index()
        )
        df = df.merge(route_stats, on="route_id", how="left")
        df["route_target_std"] = df["route_target_std"].fillna(0)
        df["route_target_rel"] = df[TARGET_COL] / (df["route_target_mean"] + 1e-6)

        office_g = df.sort_values("timestamp").groupby("office_from_id", sort=False)
        df["office_target_lag1"] = office_g["office_target_sum"].shift(1)
        df["office_target_lag48"] = office_g["office_target_sum"].shift(48)

        seasonal_enc = (
            df.groupby(["route_id", "day_of_week", "halfhour_slot"])[TARGET_COL]
            .agg(
                route_slot_mean="mean",
                route_slot_median="median",
                route_slot_std="std",
            )
            .reset_index()
        )
        df = df.merge(
            seasonal_enc,
            on=["route_id", "day_of_week", "halfhour_slot"],
            how="left",
        )
        df["route_slot_std"] = df["route_slot_std"].fillna(0)
        df["slot_ratio"] = df[TARGET_COL] / (df["route_slot_mean"] + 1e-6)

        df["target_lag_672"] = g[TARGET_COL].shift(672)

        weekly_lags = ["target_lag_336", "target_lag_672"]
        existing_weekly = [c for c in weekly_lags if c in df.columns]
        if existing_weekly:
            df["target_weekly_avg"] = df[existing_weekly].mean(axis=1)

    return df.sort_values(["route_id", "timestamp"]).reset_index(drop=True)


def add_winning_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling_cv and lag_ratio features used in exp_best_features_full."""
    for w in [4, 8, 48]:
        std_col = f"target_roll_std_{w}"
        mean_col = f"target_roll_mean_{w}"
        if std_col in df.columns and mean_col in df.columns:
            df[f"target_cv_{w}"] = df[std_col] / (df[mean_col] + 1e-6)

    if "target_lag_1" in df.columns and "target_lag_48" in df.columns:
        df["lag_ratio_1_48"] = df["target_lag_1"] / (df["target_lag_48"] + 1e-6)
    if "target_lag_1" in df.columns and "target_lag_336" in df.columns:
        df["lag_ratio_1_336"] = df["target_lag_1"] / (df["target_lag_336"] + 1e-6)
    if "target_lag_1" in df.columns and "target_ema_8" in df.columns:
        df["momentum_1_ema8"] = df["target_lag_1"] - df["target_ema_8"]
    if "target_lag_1" in df.columns and "target_ema_24" in df.columns:
        df["momentum_1_ema24"] = df["target_lag_1"] - df["target_ema_24"]

    return df


def build_feature_cols(df: pd.DataFrame) -> List[str]:
    """Build feature list exactly as in training (exclude targets, ts, id)."""
    exclude = {TARGET_COL, "timestamp", "id", *FUTURE_TARGET_COLS}
    return [c for c in df.columns if c not in exclude]


def encode_id_categoricals(X: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Convert *_id columns to pandas categorical dtype for LightGBM prediction."""
    X = X.copy()
    cat_features = [c for c in feature_cols if c.endswith("_id") and c in X.columns]
    for col in cat_features:
        X[col] = X[col].astype(str).astype("category")
    return X


def load_models(models_dir: Path) -> List[Dict[str, object]]:
    """Load all ensemble pickle files from a directory."""
    model_paths = sorted(models_dir.glob("*.pkl"))
    if not model_paths:
        raise FileNotFoundError(f"No .pkl files found in: {models_dir}")

    models = []
    for p in model_paths:
        m = joblib.load(p)
        if not isinstance(m, dict):
            raise TypeError(f"Unexpected model object in {p.name}: {type(m)}")
        for required_key in ("target_step_1", "target_step_5", "target_step_9"):
            if required_key not in m:
                raise KeyError(f"{p.name} missing key: {required_key}")
        models.append(m)
    return models


def prepare_feature_matrix(train_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """Load train data and build full feature matrix used for point-in-time inference."""
    df = pd.read_parquet(train_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["route_id", "timestamp"]).reset_index(drop=True)

    df = make_features(df, extended=True)
    df = add_winning_features(df)
    feature_cols = build_feature_cols(df)

    # route_id is already in feature_cols, so avoid duplicate column labels
    keep_cols = feature_cols + ["timestamp"]
    X_all = df[keep_cols].copy()
    return X_all, feature_cols


def predict_for_route_timestamp(
    X_all: pd.DataFrame,
    feature_cols: List[str],
    models: List[Dict[str, object]],
    route_id: str,
    timestamp: str,
) -> Dict[str, float]:
    """Predict 0-2h / 2-4h / 4-6h for one route_id and timestamp."""
    ts = pd.to_datetime(timestamp)

    row_mask = (X_all["route_id"].astype(str) == str(route_id)) & (X_all["timestamp"] == ts)
    row_df = X_all.loc[row_mask, feature_cols].copy()

    if row_df.empty:
        raise ValueError(
            f"No row found for route_id={route_id}, timestamp={ts}. "
            "Check that timestamp exists exactly in train parquet."
        )

    row_df = encode_id_categoricals(row_df, feature_cols)

    p_step1 = []
    p_step5 = []
    p_step9 = []

    for m in models:
        p_step1.append(float(np.clip(m["target_step_1"].predict(row_df)[0], 0.0, None)))
        p_step5.append(float(np.clip(m["target_step_5"].predict(row_df)[0], 0.0, None)))
        p_step9.append(float(np.clip(m["target_step_9"].predict(row_df)[0], 0.0, None)))

    out = {
        "route_id": str(route_id),
        "timestamp": str(ts),
        "n_models": len(models),
        "pred_0_2h": float(np.mean(p_step1)),
        "pred_2_4h": float(np.mean(p_step5)),
        "pred_4_6h": float(np.mean(p_step9)),
        "pred_0_2h_std": float(np.std(p_step1)),
        "pred_2_4h_std": float(np.std(p_step5)),
        "pred_4_6h_std": float(np.std(p_step9)),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone prediction for exp_best_features_full ensemble"
    )
    parser.add_argument("--route_id", required=True, help="Route id to predict for")
    parser.add_argument(
        "--timestamp",
        required=True,
        help="Timestamp to predict from (example: 2025-05-01 12:00:00)",
    )
    parser.add_argument(
        "--train_path",
        default=DEFAULT_TRAIN_PATH,
        help=f"Path to train parquet (default: {DEFAULT_TRAIN_PATH})",
    )
    parser.add_argument(
        "--models_dir",
        default=DEFAULT_MODELS_DIR,
        help=f"Directory with ensemble .pkl files (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--json_out",
        default="",
        help="Optional path to save output JSON",
    )

    args = parser.parse_args()

    train_path = Path(args.train_path)
    models_dir = Path(args.models_dir)

    if not train_path.exists():
        raise FileNotFoundError(f"train_path not found: {train_path}")
    if not models_dir.exists():
        raise FileNotFoundError(f"models_dir not found: {models_dir}")

    print("Loading models...")
    models = load_models(models_dir)
    print(f"Loaded {len(models)} ensemble members")

    print("Building features...")
    X_all, feature_cols = prepare_feature_matrix(train_path)
    print(f"Feature columns: {len(feature_cols)}")

    print("Running prediction...")
    result = predict_for_route_timestamp(
        X_all=X_all,
        feature_cols=feature_cols,
        models=models,
        route_id=args.route_id,
        timestamp=args.timestamp,
    )

    print("\nPrediction result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved JSON: {out_path}")


if __name__ == "__main__":
    main()
