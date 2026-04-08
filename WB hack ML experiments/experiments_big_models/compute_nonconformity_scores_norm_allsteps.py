"""Compute normalized non-conformity scores for ALL forecast horizons.

Non-conformity score formula:
    score = |y_actual - y_hat| / (y_hat + 1e-6)

Horizon labels (target_2h is a 2h window ending at t + k*30min):
    step_1  -> "-1.5-0.5h"
    step_2  -> "-1-1h"
    step_3  -> "-0.5-1.5h"
    step_4  -> "0-2h"
    step_5  -> "0.5-2.5h"
    step_6  -> "1-3h"
    step_7  -> "1.5-3.5h"
    step_8  -> "2-4h"
    step_9  -> "2.5-4.5h"
    step_10 -> "3-5h"
    step_11 -> "3.5-5.5h"
    step_12 -> "4-6h"

Only steps actually present in all ensemble pkl files are scored.
The validation split uses the same 21-day / 80% time-based logic as training.
Each step is scored independently on rows where its future target is non-null.

Output: data/non_conformity_scores_norm_allsteps.csv
        columns: route_id, horizon, step, score
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
OUT_PATH = Path("data/non_conformity_scores_norm_allsteps.csv")

# All possible horizons up to step 12
HORIZON_LABELS = {
    1:  "-1.5-0.5h",
    2:  "-1-1h",
    3:  "-0.5-1.5h",
    4:  "0-2h",
    5:  "0.5-2.5h",
    6:  "1-3h",
    7:  "1.5-3.5h",
    8:  "2-4h",
    9:  "2.5-4.5h",
    10: "3-5h",
    11: "3.5-5.5h",
    12: "4-6h",
}

MAX_STEP = max(HORIZON_LABELS)


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

# Capture feature_cols BEFORE adding future targets
feature_cols = build_feature_cols(train_df)
print(f"  Feature count: {len(feature_cols)}")

# Build all target_step_k columns
print(f"  Building target_step_1..{MAX_STEP}...")
sys.stdout.flush()
g = train_df.groupby("route_id", sort=False)
for k in range(1, MAX_STEP + 1):
    col = f"target_step_{k}"
    if col not in train_df.columns:
        train_df[col] = g[TARGET_COL].shift(-k)

# ── Determine validation split (based on feature rows, no dropna on targets) ──
print("Determining validation split window...")
sys.stdout.flush()

model_df = train_df[feature_cols + ["timestamp"]].rename(columns={"timestamp": "source_timestamp"}).copy()
ts_max = model_df["source_timestamp"].max()
ts_start = ts_max - pd.Timedelta(days=TRAIN_DAYS)
model_df = model_df[model_df["source_timestamp"] >= ts_start].sort_values("source_timestamp")
split_point = model_df["source_timestamp"].quantile(VALID_FRAC)

valid_idx = model_df[model_df["source_timestamp"] > split_point].index
valid_df = train_df.loc[valid_idx].copy()
valid_df_ts = model_df.loc[valid_idx, "source_timestamp"]

X_valid_full = valid_df[feature_cols].copy()
route_ids_full = valid_df["route_id"].values
print(f"  Total validation rows: {len(valid_df)}")
sys.stdout.flush()

# ── Load ensemble models ──────────────────────────────────────────────────────
print("Loading ensemble pkl files...")
sys.stdout.flush()
pkl_files = sorted(MODELS_DIR.glob("*.pkl"))
if not pkl_files:
    raise FileNotFoundError(f"No pkl files found in {MODELS_DIR}")
ensembles = [joblib.load(p) for p in pkl_files]
print(f"  Loaded {len(ensembles)} ensemble members")

# Determine which steps are available in ALL ensemble members
available_steps = [
    k for k in range(1, MAX_STEP + 1)
    if all(f"target_step_{k}" in m for m in ensembles)
]
skipped = [k for k in range(1, MAX_STEP + 1) if k not in available_steps]
print(f"  Available steps: {available_steps}")
if skipped:
    print(f"  Skipped (not trained yet): {skipped}")
sys.stdout.flush()

# Encode categoricals once for all of X_valid_full
cat_cols = [c for c in feature_cols if c.endswith("_id")]
all_cats = {col: X_valid_full[col].astype(str).unique() for col in cat_cols}
for col, cats in all_cats.items():
    X_valid_full[col] = X_valid_full[col].astype(str).astype(pd.CategoricalDtype(categories=cats))

# ── Compute normalized non-conformity scores per step ────────────────────────
print("Computing normalized non-conformity scores...")
sys.stdout.flush()

chunks = []
for k in available_steps:
    step_key = f"target_step_{k}"
    label = HORIZON_LABELS[k]

    # Filter to rows where this step's target is non-null
    target_vals = valid_df[step_key].values
    mask = ~np.isnan(target_vals)
    y_true = target_vals[mask]
    X_step = X_valid_full.iloc[mask]
    route_ids_step = route_ids_full[mask]

    # Ensemble average prediction
    preds = np.mean(
        [np.clip(m[step_key].predict(X_step), 0, None) for m in ensembles],
        axis=0,
    )

    scores = np.abs(y_true - preds) / (preds + 1e-6)

    wape = np.abs(y_true - preds).sum() / (y_true.sum() + 1e-8)
    print(f"  step_{k:2d} [{label:>12s}] n={mask.sum():>7d}  "
          f"WAPE={wape:.4f}  median={np.median(scores):.4f}  p95={np.percentile(scores, 95):.4f}")
    sys.stdout.flush()

    chunks.append(pd.DataFrame({
        "route_id": route_ids_step.astype(int),
        "horizon": label,
        "step": k,
        "score": np.round(scores, 6),
    }))

# ── Save ──────────────────────────────────────────────────────────────────────
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result_df = pd.concat(chunks, ignore_index=True)
result_df.to_csv(OUT_PATH, index=False)

print(f"\nSaved {len(result_df)} rows to {OUT_PATH}")
print(result_df.groupby(["step", "horizon"])["score"].agg(["median", lambda x: np.percentile(x, 95)]).rename(
    columns={"median": "p50", "<lambda_0>": "p95"}
).round(4).to_string())
