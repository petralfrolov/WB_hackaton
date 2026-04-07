"""Compute normalized non-conformity scores on the held-out validation set.

Non-conformity score formula:
    score = |y_actual - y_hat| / (y_hat + 1e-6)

Horizons:
    0-2h  -> target_step_4  (= target_2h at +2h boundary)
    2-4h  -> target_step_8  (= target_2h at +4h boundary)
    4-6h  -> target_step_12 (= target_2h at +6h boundary, if models exist)

The validation set is the same 20% time-based split used during training
(source_timestamp > 80th percentile of the 21-day window).

Output: data/non_conformity_scores_norm.csv  with columns: route_id, horizon, score
"""

import sys
sys.path.insert(0, ".")

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from config import TARGET_COL, TRAIN_DAYS, VALID_FRAC
from data import load_data, build_feature_cols
from features import make_features

MODELS_DIR = Path("models/exp_best_features_full")
OUT_PATH = Path("data/non_conformity_scores_norm.csv")

# Horizons: (step_key_in_pkl, horizon_label, shift_n)
HORIZONS = [
    ("target_step_4",  "0-2h",  4),
    ("target_step_8",  "2-4h",  8),
    ("target_step_12", "4-6h", 12),
]


def add_winning_features(df: pd.DataFrame) -> pd.DataFrame:
    for w in [4, 8, 48]:
        sc, mc = f"target_roll_std_{w}", f"target_roll_mean_{w}"
        if sc in df.columns and mc in df.columns:
            df[f"target_cv_{w}"] = df[sc] / (df[mc] + 1e-6)
    if "target_lag_1" in df.columns and "target_lag_48" in df.columns:
        df["lag_ratio_1_48"] = df["target_lag_1"] / (df["target_lag_48"] + 1e-6)
    if "target_lag_1" in df.columns and "target_lag_336" in df.columns:
        df["lag_ratio_1_336"] = df["target_lag_1"] / (df["target_lag_336"] + 1e-6)
    if "target_lag_1" in df.columns and "target_ema_8" in df.columns:
        df["momentum_1_ema8"] = df["target_lag_1"] - df["target_ema_8"]
    if "target_lag_1" in df.columns and "target_ema_24" in df.columns:
        df["momentum_1_ema24"] = df["target_lag_1"] - df["target_ema_24"]
    return df


# ── Load data & build features ────────────────────────────────────────────────
print("Loading data and building features...")
sys.stdout.flush()
train_df, _ = load_data()
train_df = make_features(train_df, extended=True)
train_df = add_winning_features(train_df)

# Capture feature_cols BEFORE adding future targets (so step_12 doesn't leak in)
feature_cols = build_feature_cols(train_df)

# Build future targets for all needed steps
g = train_df.groupby("route_id", sort=False)
for _, _, shift_n in HORIZONS:
    col = f"target_step_{shift_n}"
    if col not in train_df.columns:
        train_df[col] = g[TARGET_COL].shift(-shift_n)
print(f"  Feature count: {len(feature_cols)}")
sys.stdout.flush()

# ── Recreate validation split (identical to split_data logic) ─────────────────
print("Recreating validation split...")
sys.stdout.flush()

step_cols_needed = [f"target_step_{s}" for _, _, s in HORIZONS]
supervised_df = train_df.dropna(subset=step_cols_needed).copy()

cols_for_model = feature_cols + ["timestamp"] + step_cols_needed
model_df = supervised_df[cols_for_model].rename(columns={"timestamp": "source_timestamp"})

ts_max = model_df["source_timestamp"].max()
ts_start = ts_max - pd.Timedelta(days=TRAIN_DAYS)
model_df = model_df[model_df["source_timestamp"] >= ts_start].sort_values("source_timestamp")

split_point = model_df["source_timestamp"].quantile(VALID_FRAC)
valid_df = model_df[model_df["source_timestamp"] > split_point].copy()

print(f"  Validation rows: {len(valid_df)}")
print(f"  Validation routes: {valid_df['route_id'].nunique()}")
sys.stdout.flush()

X_valid = valid_df[feature_cols].copy()
route_ids_valid = valid_df["route_id"].values

# ── Load ensemble models ──────────────────────────────────────────────────────
print("Loading ensemble pkl files...")
sys.stdout.flush()
pkl_files = sorted(MODELS_DIR.glob("*.pkl"))
if not pkl_files:
    raise FileNotFoundError(f"No pkl files found in {MODELS_DIR}")

ensembles = [joblib.load(p) for p in pkl_files]
print(f"  Loaded {len(ensembles)} ensemble members")

# Check which horizons are actually available
available_horizons = [
    (step_key, label, shift_n)
    for step_key, label, shift_n in HORIZONS
    if all(step_key in m for m in ensembles)
]
skipped = set(label for _, label, _ in HORIZONS) - set(label for _, label, _ in available_horizons)
if skipped:
    print(f"  Skipped horizons (models not trained yet): {skipped}")
sys.stdout.flush()

# Encode categoricals for prediction
cat_cols = [c for c in feature_cols if c.endswith("_id")]
all_cats = {col: X_valid[col].astype(str).unique() for col in cat_cols}
for col, cats in all_cats.items():
    X_valid[col] = X_valid[col].astype(str).astype(pd.CategoricalDtype(categories=cats))

# ── Compute normalized non-conformity scores per horizon ─────────────────────
print("Computing normalized non-conformity scores...")
sys.stdout.flush()

chunks = []
for step_key, label, shift_n in available_horizons:
    y_true = valid_df[f"target_step_{shift_n}"].values

    # Ensemble average prediction
    preds = np.mean(
        [np.clip(m[step_key].predict(X_valid), 0, None) for m in ensembles],
        axis=0,
    )

    scores = np.abs(y_true - preds) / (preds + 1e-6)

    n = len(scores)
    w_mean = np.abs(y_true - preds).sum() / (y_true.sum() + 1e-8)
    print(f"  {label}: n={n}  WAPE={w_mean:.4f}  "
          f"score mean={scores.mean():.4f}  median={np.median(scores):.4f}  "
          f"p95={np.percentile(scores, 95):.4f}")
    sys.stdout.flush()

    chunks.append(pd.DataFrame({
        "route_id": route_ids_valid.astype(int),
        "horizon": label,
        "score": np.round(scores, 6),
    }))

# ── Save ──────────────────────────────────────────────────────────────────────
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result_df = pd.concat(chunks, ignore_index=True)
result_df.to_csv(OUT_PATH, index=False)

print(f"\nSaved {len(result_df)} rows to {OUT_PATH}")
print(result_df.groupby("horizon")["score"].describe().round(6).to_string())
