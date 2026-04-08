"""Standalone inference pipeline for exp_best_features_full.

Produces three forecasts for a selected route_id and timestamp using the saved
ensemble of LightGBM step models.

Target column is target_2h — a 2-hour rolling window value.  Each step is
30 minutes, so to predict the 2-hour window anchored exactly at a horizon
boundary we pick the step at that boundary:
  step_4  = target_2h in 4×30min = exactly +2h ahead  (0–2h window)
  step_8  = target_2h in 8×30min = exactly +4h ahead  (2–4h window)
  step_12 = target_2h in 12×30min = exactly +6h ahead (4–6h window)
             — requires models trained beyond step 10; returns None if absent.

This file is fully independent from project modules (no local imports).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb  # noqa: F401 - needed for unpickling LGB models
import numpy as np
import pandas as pd

from config import (
    DEFAULT_DEPRECATED_MODELS_DIR as CONFIG_DEFAULT_DEPRECATED_MODELS_DIR,
    DEFAULT_TRAIN_PATH as CONFIG_DEFAULT_TRAIN_PATH,
    FORECAST_POINTS as CONFIG_FORECAST_POINTS,
    HALFHOUR_SLOT_MINUTES,
    LOOKBACK_HEADROOM_DAYS,
    MAX_LAG_STEPS as CONFIG_MAX_LAG_STEPS,
    TARGET_COLUMN,
    TRACK_NAME,
)

# Hardcoded config (team track)
# -----------------------------
TRACK = TRACK_NAME
TARGET_COL = TARGET_COLUMN
FORECAST_POINTS = CONFIG_FORECAST_POINTS
FUTURE_TARGET_COLS = [f"target_step_{i}" for i in range(1, FORECAST_POINTS + 1)]

DEFAULT_TRAIN_PATH = str(CONFIG_DEFAULT_TRAIN_PATH)
DEFAULT_MODELS_DIR = str(CONFIG_DEFAULT_DEPRECATED_MODELS_DIR)


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar time features: hour, day-of-week, cyclic encodings, half-hour slot."""
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.int8)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["halfhour_slot"] = df["hour"] * 2 + df["timestamp"].dt.minute // HALFHOUR_SLOT_MINUTES
    return df


def _add_lag_rolling_features(df: pd.DataFrame, extended: bool) -> pd.DataFrame:
    """Add target lags, rolling statistics (mean/std/min/max), EMA, and first-differences."""
    g = df.groupby("route_id", sort=False)

    base_lags = [1, 2, 4, 8, 16, 48]
    if extended:
        base_lags += [96, 192, 336]
    for lag in base_lags:
        df[f"target_lag_{lag}"] = g[TARGET_COL].shift(lag)

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

    df["target_diff_1"] = g[TARGET_COL].diff(1)
    df["target_diff_2"] = g[TARGET_COL].diff(2)
    df["target_diff_4"] = g[TARGET_COL].diff(4)

    for w in [8, 48]:
        df[f"target_roll_min_{w}"] = g[TARGET_COL].transform(
            lambda x, w=w: x.shift(1).rolling(w, min_periods=1).min()
        )
        df[f"target_roll_max_{w}"] = g[TARGET_COL].transform(
            lambda x, w=w: x.shift(1).rolling(w, min_periods=1).max()
        )
    return df


def _add_status_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add status-column lags, rolling means, ratio encodings, and summary stats.

    Only executed when ``status_*`` columns are present in ``df``.
    """
    status_cols = sorted([
        c for c in df.columns
        if c.startswith("status_")
        and not c.endswith("_ratio")
        and "_lag" not in c
        and "_roll" not in c
    ])
    if not status_cols:
        return df

    g = df.groupby("route_id", sort=False)
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
    return df


def _add_deconvolution_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deconvolution features s_t0..s_t3 by summing alternating-sign lagged targets.

    Each s_ti is a clipped running difference that approximates a discrete-time
    deconvolution of the demand signal over 12 periods.
    """
    g = df.groupby("route_id", sort=False)
    n_deconv_pairs = 12
    extra_lag_cols: dict = {}
    deconv_cols: dict = {}
    for offset, name in [(0, "s_t0"), (1, "s_t1"), (2, "s_t2"), (3, "s_t3")]:
        terms = []
        for k in range(n_deconv_pairs):
            lag_add = offset + 4 * k
            lag_sub = offset + 4 * k + 1
            lag_add_name = f"target_lag_{lag_add}"
            lag_sub_name = f"target_lag_{lag_sub}"
            if lag_add_name not in df.columns and lag_add_name not in extra_lag_cols:
                extra_lag_cols[lag_add_name] = g[TARGET_COL].shift(lag_add)
            if lag_sub_name not in df.columns and lag_sub_name not in extra_lag_cols:
                extra_lag_cols[lag_sub_name] = g[TARGET_COL].shift(lag_sub)
            lag_add_s = df[lag_add_name] if lag_add_name in df.columns else extra_lag_cols[lag_add_name]
            lag_sub_s = df[lag_sub_name] if lag_sub_name in df.columns else extra_lag_cols[lag_sub_name]
            terms.append(lag_add_s)
            terms.append(-lag_sub_s)
        s = terms[0]
        for t in terms[1:]:
            s = s + t
        deconv_cols[f"deconv_{name}"] = s.clip(lower=0)

    new_cols = {**extra_lag_cols, **deconv_cols}
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def _add_office_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add office-level aggregated target statistics (mean, sum, std) via left merge."""
    office_agg = (
        df.groupby(["office_from_id", "timestamp"])[TARGET_COL]
        .agg(office_target_mean="mean", office_target_sum="sum", office_target_std="std")
        .reset_index()
    )
    df = df.merge(office_agg, on=["office_from_id", "timestamp"], how="left")
    df["office_target_std"] = df["office_target_std"].fillna(0)
    return df


def _add_extended_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add route-level encoding, seasonal slot stats, office lags, and weekly lags.

    Requires ``office_target_sum``, ``day_of_week``, and ``halfhour_slot`` columns
    to already exist in ``df`` (added by earlier helpers).
    """
    g = df.groupby("route_id", sort=False)

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
        .agg(route_slot_mean="mean", route_slot_median="median", route_slot_std="std")
        .reset_index()
    )
    df = df.merge(seasonal_enc, on=["route_id", "day_of_week", "halfhour_slot"], how="left")
    df["route_slot_std"] = df["route_slot_std"].fillna(0)
    df["slot_ratio"] = df[TARGET_COL] / (df["route_slot_mean"] + 1e-6)

    df["target_lag_672"] = g[TARGET_COL].shift(672)
    weekly_lags = ["target_lag_336", "target_lag_672"]
    existing_weekly = [c for c in weekly_lags if c in df.columns]
    if existing_weekly:
        df["target_weekly_avg"] = df[existing_weekly].mean(axis=1)
    return df


def make_features(df: pd.DataFrame, extended: bool = True) -> pd.DataFrame:
    """Replicate training feature engineering used in experiments.

    Applies the full feature pipeline in order:
    time features → lags/rolling → status → deconvolution → office agg → extended.
    The ``extended`` flag enables longer lags, EMA-96, and route/seasonal encodings
    used in the exp_best_features_full ensemble.
    """
    df = df.sort_values(["route_id", "timestamp"]).copy()
    df = _add_time_features(df)
    df = _add_lag_rolling_features(df, extended)
    df = _add_status_features(df)
    df = _add_deconvolution_features(df)
    df = _add_office_features(df)
    if extended:
        df = _add_extended_features(df)
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
        # step_4 (+2h) and step_8 (+4h) are required; step_12 (+6h) is optional
        # (needs to be trained separately — only 10 steps exist in current ensembles)
        for required_key in ("target_step_4", "target_step_8"):
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
    """Predict all step horizons for one route_id and timestamp.

    Returns per-step ensemble-averaged predictions (pred_step_1 through
    pred_step_12) plus backward-compatible named outputs:
      pred_0_2h  = step_4  (0-2h window)
      pred_2_4h  = step_8  (2-4h window)
      pred_4_6h  = step_12 (4-6h window)
    """
    ts = pd.to_datetime(timestamp)

    row_mask = (X_all["route_id"].astype(str) == str(route_id)) & (X_all["timestamp"] == ts)
    row_df = X_all.loc[row_mask, feature_cols].copy()

    if row_df.empty:
        raise ValueError(
            f"No row found for route_id={route_id}, timestamp={ts}. "
            "Check that timestamp exists exactly in train parquet."
        )

    row_df = encode_id_categoricals(row_df, feature_cols)

    # Discover step keys common to ALL ensemble members
    step_keys = sorted(
        {k for k in models[0] if k.startswith("target_step_")}
        & set.intersection(*(set(m) for m in models))
    )

    # Predict every available step across all ensemble members
    step_preds: Dict[str, List[float]] = {k: [] for k in step_keys}
    for m in models:
        for k in step_keys:
            step_preds[k].append(float(np.clip(m[k].predict(row_df)[0], 0.0, None)))

    out: Dict[str, object] = {
        "route_id": str(route_id),
        "timestamp": str(ts),
        "n_models": len(models),
    }

    # Per-step ensemble averages
    for k, vals in step_preds.items():
        step_num = int(k.replace("target_step_", ""))
        out[f"pred_step_{step_num}"] = float(round(np.mean(vals)))
        out[f"pred_step_{step_num}_std"] = float(np.std(vals))

    # Backward-compatible named outputs
    has_step12 = "target_step_12" in step_keys
    out["pred_0_2h"] = out.get("pred_step_4", 0.0)
    out["pred_0_2h_std"] = out.get("pred_step_4_std", 0.0)
    out["pred_2_4h"] = out.get("pred_step_8", 0.0)
    out["pred_2_4h_std"] = out.get("pred_step_8_std", 0.0)
    out["pred_4_6h"] = out.get("pred_step_12") if has_step12 else None
    out["pred_4_6h_std"] = out.get("pred_step_12_std") if has_step12 else None
    out["pred_4_6h_available"] = has_step12
    return out


# ---------------------------------------------------------------------------
# Lazy inference: load only the required parquet window per request
# ---------------------------------------------------------------------------

#: Max lag used in feature engineering (target_lag_672 = 672 × 30 min = 14 days)
_MAX_LAG_STEPS = CONFIG_MAX_LAG_STEPS
_LAG_MINUTES = _MAX_LAG_STEPS * 30  # 20 160 min ≈ 14 days
_LOOKBACK_DAYS = _LAG_MINUTES // (60 * 24) + LOOKBACK_HEADROOM_DAYS


def prepare_feature_matrix_for_route(
    train_path: Path,
    route_id: str,
    timestamp: str,
    office_routes: Optional[List[str]] = None,
) -> Tuple["pd.DataFrame", List[str]]:
    """Load a small time window for *route_id* (and same-office peers) and build features.

    Only reads rows where route_id is in the relevant set and timestamp is within
    the lookback window — much cheaper than loading the full parquet at startup.
    """
    ts = pd.to_datetime(timestamp)
    min_ts = ts - pd.Timedelta(days=_LOOKBACK_DAYS)

    routes_to_load = list({str(route_id)} | {str(r) for r in (office_routes or [])})

    # Predicate pushdown: prune row-groups by route_id when dtype is compatible.
    # Some parquet builds store route_id as int64; passing string filters then crashes
    # inside pyarrow (e.g. ArrowNotImplementedError: string vs int64 compare).
    # Try int filter first (covers int64-stored parquet), then string, then full read.
    int_routes = []
    for r in routes_to_load:
        try:
            int_routes.append(int(r))
        except (ValueError, TypeError):
            pass

    df = None
    if int_routes:
        try:
            df = pd.read_parquet(train_path, filters=[("route_id", "in", int_routes)])
            # Verify the filter actually returned rows (empty means wrong dtype)
            if df.empty:
                df = None
        except Exception as e:
            logger.debug("int filter failed for route_id=%s, falling back to string filter: %s", route_id, e)
            df = None

    if df is None:
        try:
            df = pd.read_parquet(train_path, filters=[("route_id", "in", routes_to_load)])
        except Exception as e:
            # Last-resort fallback: full read. Slower but dtype-agnostic.
            logger.warning(
                "Parquet predicate pushdown failed for route_id=%s (dtype mismatch?), "
                "falling back to full parquet read — this is slow under load. Error: %s",
                route_id, e,
            )
            df = pd.read_parquet(train_path)

    # Normalize key columns before filtering.
    df["route_id"] = df["route_id"].astype(str)
    df = df[df["route_id"].isin(routes_to_load)]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df[df["timestamp"].notna()]
    df = df[(df["timestamp"] >= min_ts) & (df["timestamp"] <= ts)]
    df = df.sort_values(["route_id", "timestamp"]).reset_index(drop=True)

    if df.empty or str(route_id) not in df["route_id"].astype(str).values:
        raise ValueError(
            f"No data found for route_id={route_id} in window [{min_ts}, {ts}]. "
            "Verify that the timestamp exists in the training parquet."
        )

    df = make_features(df, extended=True)
    df = add_winning_features(df)
    feature_cols = build_feature_cols(df)

    keep = feature_cols + ["timestamp"]
    return df[keep].copy(), feature_cols


def predict_lazy(
    train_path: Path,
    models: List[Dict[str, object]],
    route_id: str,
    timestamp: str,
    office_routes: Optional[List[str]] = None,
) -> Dict[str, float]:
    """End-to-end lazy prediction: read parquet window → build features → ensemble predict.

    Uses the same ``predict_for_route_timestamp`` logic but avoids pre-loading
    the entire feature matrix at application startup.
    """
    X_all, feature_cols = prepare_feature_matrix_for_route(
        train_path, route_id, timestamp, office_routes
    )
    return predict_for_route_timestamp(X_all, feature_cols, models, route_id, timestamp)


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
