"""Experiment 30: Enhanced feature engineering on full data.

Built on exp19 framework (10 seeds, alpha=0.55, 12-pair deconv).
New features added on top of make_features(extended=True):
  1. Rolling coefficient of variation (CV = std/mean) — routes with high CV
     need different predictions; captures demand volatility.
  2. Fourier harmonics for hour-of-day and day-of-week (2nd and 3rd harmonics)
     — captures smoother intra-day/weekly transitions than one sin/cos pair.
  3. Cross-route warehouse aggregates — total volume per (office_to_id, timestamp),
     giving model global demand context beyond individual route.
  4. Time-since-midnight continuous feature + interaction with day_of_week.
  5. Route busyness rank within its office (where does this route sit vs peers).
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, ".")

from config import TRACK, FUTURE_TARGET_COLS, TARGET_COL
from data import load_data, create_future_targets, build_feature_cols, split_data, encode_categoricals
from features import make_features
from metrics import WapePlusRbias
from train import train_lgb_models, predict_steps, build_submission

EXPERIMENT_NAME = "exp30_enhanced_features_full"
SEEDS = [42, 123, 456, 789, 1234, 2024, 7, 314, 99, 888]

BASE_LGB = dict(
    n_estimators=5000, learning_rate=0.05, num_leaves=127,
    min_child_samples=20, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    objective="quantile", alpha=0.55,
    n_jobs=-1, verbose=-1,
)

def add_enhanced_features(df):
    """Add new features on top of base make_features output."""
    g = df.groupby("route_id", sort=False)

    # ── 1. Rolling coefficient of variation ────────────────────────────────────
    for w in [4, 8, 48]:
        mean_col = f"target_roll_mean_{w}"
        std_col = f"target_roll_std_{w}"
        if mean_col in df.columns and std_col in df.columns:
            df[f"target_cv_{w}"] = df[std_col] / (df[mean_col] + 1e-6)

    # ── 2. Higher Fourier harmonics ────────────────────────────────────────────
    hour = df["hour"]
    dow = df["day_of_week"]
    for k in [2, 3]:
        df[f"hour_sin_{k}"] = np.sin(2 * np.pi * k * hour / 24)
        df[f"hour_cos_{k}"] = np.cos(2 * np.pi * k * hour / 24)
        df[f"dow_sin_{k}"] = np.sin(2 * np.pi * k * dow / 7)
        df[f"dow_cos_{k}"] = np.cos(2 * np.pi * k * dow / 7)

    # Half-hour slot Fourier (period=48, captures sub-daily patterns)
    slot = df["halfhour_slot"]
    for k in [1, 2, 3]:
        df[f"slot_sin_{k}"] = np.sin(2 * np.pi * k * slot / 48)
        df[f"slot_cos_{k}"] = np.cos(2 * np.pi * k * slot / 48)

    # ── 3. Cross-route destination warehouse aggregates ────────────────────────
    if "office_to_id" in df.columns:
        dest_agg = (
            df.groupby(["office_to_id", "timestamp"])[TARGET_COL]
            .agg(dest_target_mean="mean", dest_target_sum="sum")
            .reset_index()
        )
        df = df.merge(dest_agg, on=["office_to_id", "timestamp"], how="left")

    # ── 4. Time-of-day continuous + interaction ────────────────────────────────
    df["time_of_day"] = df["hour"] + df["timestamp"].dt.minute / 60.0
    df["dow_x_time"] = df["day_of_week"] * 48 + df["halfhour_slot"]

    # ── 5. Route busyness rank within office ───────────────────────────────────
    if "route_target_mean" in df.columns:
        office_route_rank = (
            df.groupby("office_from_id")["route_target_mean"]
            .rank(pct=True)
        )
        df["route_office_rank"] = office_route_rank

    # ── 6. Lag ratios (current vs recent history) ──────────────────────────────
    if "target_lag_1" in df.columns and "target_lag_48" in df.columns:
        df["lag_ratio_1_48"] = df["target_lag_1"] / (df["target_lag_48"] + 1e-6)
    if "target_lag_1" in df.columns and "target_lag_336" in df.columns:
        df["lag_ratio_1_336"] = df["target_lag_1"] / (df["target_lag_336"] + 1e-6)
    # Recent momentum: lag1 vs ema
    if "target_lag_1" in df.columns and "target_ema_8" in df.columns:
        df["momentum_1_ema8"] = df["target_lag_1"] - df["target_ema_8"]
    if "target_lag_1" in df.columns and "target_ema_24" in df.columns:
        df["momentum_1_ema24"] = df["target_lag_1"] - df["target_ema_24"]

    return df


print("=" * 70)
print(f"EXPERIMENT: {EXPERIMENT_NAME}")
print(f"  Full data | 10 seeds | alpha=0.55 | enhanced features")
print("=" * 70)
sys.stdout.flush()

# ── Data ───────────────────────────────────────────────────────────────────────
print("Loading data...")
sys.stdout.flush()
train_df, test_df = load_data()
print(f"Routes: {train_df['route_id'].nunique()} | Train rows: {len(train_df)}")
sys.stdout.flush()

print("Building base features (extended + 12-pair deconv)...")
sys.stdout.flush()
train_df = make_features(train_df, extended=True)

print("Adding enhanced features...")
sys.stdout.flush()
train_df = add_enhanced_features(train_df)

train_df = create_future_targets(train_df)
feature_cols = build_feature_cols(train_df)
print(f"Feature count: {len(feature_cols)}")
sys.stdout.flush()

X_fit, y_fit, X_valid, y_valid, X_test, inference_ts, test_model_df = split_data(train_df, feature_cols)
X_fit, X_valid, X_test, cat_features = encode_categoricals(X_fit, X_valid, X_test, feature_cols)

metric = WapePlusRbias()
models_dir = Path("models") / EXPERIMENT_NAME
models_dir.mkdir(parents=True, exist_ok=True)

valid_preds_list = []
test_preds_list = []

# ── Train 10 seeds ─────────────────────────────────────────────────────────────
for seed in SEEDS:
    t0 = datetime.now()
    print(f"\n--- Seed {seed} (started {t0.strftime('%H:%M:%S')}) ---")
    sys.stdout.flush()
    params = {**BASE_LGB, "random_state": seed}
    models = train_lgb_models(X_fit, y_fit, X_valid, y_valid, lgb_params=params)
    vp = predict_steps(models, X_valid)
    tp = predict_steps(models, X_test)
    w, r, t = metric.calculate_components(y_valid, vp)
    elapsed = (datetime.now() - t0).seconds
    print(f"  Seed {seed}: WAPE={w:.4f} RBias={r:.4f} Total={t:.4f}  [{elapsed}s]")
    sys.stdout.flush()
    valid_preds_list.append(vp)
    test_preds_list.append(tp)
    joblib.dump(models, models_dir / f"lgb_seed{seed}.pkl")

# ── Ensemble ───────────────────────────────────────────────────────────────────
print("\n[Ensemble] Averaging 10 predictions...")
sys.stdout.flush()

valid_ens = pd.DataFrame(
    np.mean([p.values for p in valid_preds_list], axis=0),
    columns=FUTURE_TARGET_COLS, index=X_valid.index,
).clip(lower=0)

test_ens = pd.DataFrame(
    np.mean([p.values for p in test_preds_list], axis=0),
    columns=FUTURE_TARGET_COLS, index=X_test.index,
).clip(lower=0)

w_raw, r_raw, t_raw = metric.calculate_components(y_valid, valid_ens)
print(f"[Ensemble raw]  WAPE={w_raw:.4f} RBias={r_raw:.4f} Total={t_raw:.4f}")
sys.stdout.flush()

# ── Per-step calibration ───────────────────────────────────────────────────────
print("\n[Per-step calibration]")
valid_corrected = valid_ens.copy()
test_corrected = test_ens.copy()
for col in FUTURE_TARGET_COLS:
    true_sum = float(y_valid[col].sum())
    pred_sum = float(valid_ens[col].sum())
    f = true_sum / pred_sum if pred_sum > 0 else 1.0
    valid_corrected[col] = (valid_ens[col] * f).clip(lower=0)
    test_corrected[col] = (test_ens[col] * f).clip(lower=0)
    print(f"  {col}: factor={f:.4f}")
sys.stdout.flush()

w_c, r_c, t_c = metric.calculate_components(y_valid, valid_corrected)
print(f"\n[Ensemble per-step] WAPE={w_c:.4f} RBias={r_c:.4f} Total={t_c:.4f}")
sys.stdout.flush()

# Also compute global bias correction for comparison
from train import compute_bias_factor
global_bf = compute_bias_factor(y_valid, valid_ens)
valid_global = (valid_ens * global_bf).clip(lower=0)
w_g, r_g, t_g = metric.calculate_components(y_valid, valid_global)
print(f"[Ensemble global]  WAPE={w_g:.4f} RBias={r_g:.4f} Total={t_g:.4f}  (factor={global_bf:.4f})")
sys.stdout.flush()

# Use whichever is better
if t_c <= t_g:
    print("Using per-step calibration (better).")
    final_test = test_corrected
    final_metric = t_c
else:
    print("Using global bias correction (better).")
    final_test = (test_ens * global_bf).clip(lower=0)
    final_metric = t_g
sys.stdout.flush()

# ── Submission ─────────────────────────────────────────────────────────────────
sub = build_submission(final_test, X_test, inference_ts, test_df, bias_factor=1.0)
sub_path = f"submission_{TRACK}_{EXPERIMENT_NAME}.csv"
sub.to_csv(sub_path, index=False)
print(f"\nSaved: {sub_path}  ({len(sub)} rows)")
sys.stdout.flush()

# ── Log ────────────────────────────────────────────────────────────────────────
log_path = Path("experiments.json")
history = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
history.append({
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "name": EXPERIMENT_NAME,
    "params": dict(
        seeds=SEEDS, n_models=10, alpha=0.55,
        features="enhanced: CV, fourier, cross-route, lag_ratios, momentum",
    ),
    "wape": round(w_c, 6),
    "rbias": round(r_c, 6),
    "total": round(final_metric, 6),
})
log_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nDONE. Best VALID total = {final_metric:.6f}")
sys.stdout.flush()
