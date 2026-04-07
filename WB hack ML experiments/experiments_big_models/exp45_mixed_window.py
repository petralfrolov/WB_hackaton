"""Experiment 45: Mixed-window ensemble — 5 models on 21 days + 5 on 35 days.

Hypothesis:
  exp40 (21-day, leaves=511) -> 0.251 public (best)
  exp44 (35-day, leaves=511) -> ? (submitted, pending public result)

  Different training windows give models exposure to different seasonal/temporal
  slices of each route's history. Averaging their predictions may add diversity
  that neither individual window captures alone, potentially improving OOD.

  All 10 models use leaves=511 alpha=0.55 (best single-window config).
  Seeds split: 21-day gets [42, 123, 456, 789, 1234]
               35-day gets [2024, 7, 314, 99, 888]
  Total = 10 models (within the limit).
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

import config as _cfg

from config import FUTURE_TARGET_COLS
from data import (
    build_feature_cols,
    create_future_targets,
    encode_categoricals,
    load_data,
    split_data,
)
from features import make_features
from metrics import WapePlusRbias
from train import build_submission, predict_steps


EXPERIMENT_NAME = "exp45_mixed_window"
SEEDS_21 = [42, 123, 456, 789, 1234]   # trained on 21-day window
SEEDS_35 = [2024, 7, 314, 99, 888]     # trained on 35-day window
ALPHA = 0.55

BASE_LGB = dict(
    n_estimators=5000,
    learning_rate=0.05,
    num_leaves=511,
    min_child_samples=10,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_jobs=-1,
    verbose=-1,
)


def add_winning_features(df):
    for w in [4, 8, 48]:
        std_col, mean_col = f"target_roll_std_{w}", f"target_roll_mean_{w}"
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


def train_lgb_models(X_fit, y_fit, X_valid, y_valid, params):
    models = {}
    for step_col in FUTURE_TARGET_COLS:
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_fit,
            y_fit[step_col],
            eval_set=[(X_valid, y_valid[step_col])],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        models[step_col] = model
        best_score = next(iter(model.best_score_.get("valid_0", {}).values()), float("nan"))
        print(f"    {step_col:20s}  iter={model.best_iteration_:4d}  score={best_score:.4f}")
        sys.stdout.flush()
    return models


print("=" * 60)
print(f"EXPERIMENT: {EXPERIMENT_NAME}")
print(f"  5 seeds x 21-day window + 5 seeds x 35-day window")
print(f"  num_leaves=511 | alpha={ALPHA} | 10 models total")
print("  Hypothesis: window diversity on top of best capacity setting")
print("=" * 60)
sys.stdout.flush()


# == Part 1: Load data once (features are window-independent) =================
print("Loading data...")
sys.stdout.flush()
train_df_raw, test_df = load_data()
print(f"Routes: {train_df_raw['route_id'].nunique()} | Train rows: {len(train_df_raw)}")
sys.stdout.flush()

print("Building features (shared)...")
sys.stdout.flush()
train_df_feat = make_features(train_df_raw, extended=True)
train_df_feat = add_winning_features(train_df_feat)
train_df_feat = create_future_targets(train_df_feat)
feature_cols = build_feature_cols(train_df_feat)
print(f"Feature count: {len(feature_cols)}")
sys.stdout.flush()

# Shared X_test (always same inference timestamp regardless of window)
_cfg.TRAIN_DAYS = 21  # set to get X_test from the right split
_, _, _, _, X_test_shared, inference_ts, test_model_df = split_data(train_df_feat, feature_cols)
X_test_enc = None  # will be encoded below

metric = WapePlusRbias()
models_dir = Path("models") / EXPERIMENT_NAME
models_dir.mkdir(parents=True, exist_ok=True)

all_valid_preds = []
all_test_preds = []


# == Part 2: Train 5 seeds on 21-day window ===================================
print("\n>>> WINDOW = 21 days <<<")
sys.stdout.flush()
_cfg.TRAIN_DAYS = 21
X_fit_21, y_fit_21, X_valid_21, y_valid_21, X_test_21, _, _ = split_data(train_df_feat, feature_cols)
X_fit_21, X_valid_21, X_test_21, _ = encode_categoricals(X_fit_21, X_valid_21, X_test_21, feature_cols)

valid_preds_21 = []
test_preds_21 = []

for seed in SEEDS_21:
    t0 = datetime.now()
    print(f"\n--- 21d leaves=511 alpha={ALPHA} seed={seed} [{t0.strftime('%H:%M:%S')}] ---")
    sys.stdout.flush()
    params = {**BASE_LGB, "objective": "quantile", "alpha": ALPHA, "random_state": seed}
    models = train_lgb_models(X_fit_21, y_fit_21, X_valid_21, y_valid_21, params)
    vp = predict_steps(models, X_valid_21)
    tp = predict_steps(models, X_test_21)
    w, r, t = metric.calculate_components(y_valid_21, vp)
    elapsed = (datetime.now() - t0).seconds
    print(f"  seed={seed} 21d: WAPE={w:.4f} RBias={r:.4f} Total={t:.4f}  [{elapsed}s]")
    sys.stdout.flush()
    valid_preds_21.append(vp)
    test_preds_21.append(tp)
    joblib.dump(models, models_dir / f"lgb_21d_seed{seed}.pkl")

# Report ensemble for 21-day models
ens_21_valid = pd.DataFrame(
    np.mean([p.values for p in valid_preds_21], axis=0),
    columns=FUTURE_TARGET_COLS, index=X_valid_21.index,
).clip(lower=0)
w_21, r_21, t_21 = metric.calculate_components(y_valid_21, ens_21_valid)
print(f"\n[21d 5-seed raw ensemble] WAPE={w_21:.4f} RBias={r_21:.4f} Total={t_21:.4f}")
sys.stdout.flush()


# == Part 3: Train 5 seeds on 35-day window ===================================
print("\n>>> WINDOW = 35 days <<<")
sys.stdout.flush()
_cfg.TRAIN_DAYS = 35
X_fit_35, y_fit_35, X_valid_35, y_valid_35, X_test_35, _, _ = split_data(train_df_feat, feature_cols)
X_fit_35, X_valid_35, X_test_35, _ = encode_categoricals(X_fit_35, X_valid_35, X_test_35, feature_cols)

valid_preds_35 = []
test_preds_35 = []

for seed in SEEDS_35:
    t0 = datetime.now()
    print(f"\n--- 35d leaves=511 alpha={ALPHA} seed={seed} [{t0.strftime('%H:%M:%S')}] ---")
    sys.stdout.flush()
    params = {**BASE_LGB, "objective": "quantile", "alpha": ALPHA, "random_state": seed}
    models = train_lgb_models(X_fit_35, y_fit_35, X_valid_35, y_valid_35, params)
    vp = predict_steps(models, X_valid_35)
    tp = predict_steps(models, X_test_35)
    w, r, t = metric.calculate_components(y_valid_35, vp)
    elapsed = (datetime.now() - t0).seconds
    print(f"  seed={seed} 35d: WAPE={w:.4f} RBias={r:.4f} Total={t:.4f}  [{elapsed}s]")
    sys.stdout.flush()
    valid_preds_35.append(vp)
    test_preds_35.append(tp)
    joblib.dump(models, models_dir / f"lgb_35d_seed{seed}.pkl")


# == Part 4: Blend test predictions ===========================================
# Note: X_test_21 and X_test_35 have same rows (same inference timestamp),
# just encoded independently. Test preds are already computed above.
print(f"\n[Ensemble] Blending {len(test_preds_21)+len(test_preds_35)} test predictions...")
sys.stdout.flush()

test_ens = pd.DataFrame(
    np.mean([p.values for p in test_preds_21 + test_preds_35], axis=0),
    columns=FUTURE_TARGET_COLS,
    index=X_test_21.index,
).clip(lower=0)

# For validation ensemble report, use 21-day val (the standard one) with raw ensemble
# We can only do apples-to-apples on 21-day val for the calibration step
valid_ens_21 = pd.DataFrame(
    np.mean([p.values for p in valid_preds_21], axis=0),
    columns=FUTURE_TARGET_COLS, index=X_valid_21.index,
).clip(lower=0)

w_raw, r_raw, t_raw = metric.calculate_components(y_valid_21, valid_ens_21)
print(f"[21d val, 5-model raw] WAPE={w_raw:.4f} RBias={r_raw:.4f} Total={t_raw:.4f}")

# Calibration based on 21-day val (same reference as all other exps)
factors = {}
test_cal = test_ens.copy()
for col in FUTURE_TARGET_COLS:
    # compute calibration factors from 21d val ensemble
    f = float(y_valid_21[col].sum()) / max(float(valid_ens_21[col].sum()), 1e-9)
    test_cal[col] = (test_ens[col] * f).clip(lower=0)
    factors[col] = round(f, 4)

print(f"[Cal factors from 21d val]: {list(factors.values())}")
sys.stdout.flush()

# Apply same factors to get calibrated val for reporting
valid_cal_21 = valid_ens_21.copy()
for col in FUTURE_TARGET_COLS:
    valid_cal_21[col] = (valid_ens_21[col] * factors[col]).clip(lower=0)

w_cal, r_cal, t_cal = metric.calculate_components(y_valid_21, valid_cal_21)
print(f"[21d val calibrated 5-model] WAPE={w_cal:.4f} RBias={r_cal:.4f} Total={t_cal:.4f}")
print("(Note: test predictions include 5x35d models — val metric above is partial indicator only)")
sys.stdout.flush()

submission = build_submission(test_cal, X_test_21, inference_ts, test_df)
submission_path = f"submission_team_{EXPERIMENT_NAME}.csv"
submission.to_csv(submission_path, index=False)
print(f"Saved: {submission_path}  ({len(submission)} rows)")
sys.stdout.flush()

experiments_path = Path("experiments.json")
experiments = json.loads(experiments_path.read_text(encoding="utf-8"))
experiments.append(
    {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "name": EXPERIMENT_NAME,
        "params": {
            "seeds_21d": SEEDS_21,
            "seeds_35d": SEEDS_35,
            "num_leaves": 511,
            "min_child_samples": 10,
            "n_models": 10,
            "features": "extended+rolling_cv+lag_ratios (154)",
            "calibration": "per_step from 21d val",
        },
        "wape": round(w_cal, 6),
        "rbias": round(r_cal, 6),
        "total": round(t_cal, 6),
        "note": (
            f"Mixed window: 5x21d + 5x35d, leaves=511 alpha={ALPHA}. "
            f"21d val 5-model raw WAPE={w_raw:.4f} RBias={r_raw:.4f}. Cal partial: {t_cal:.4f}. "
            f"Factors from 21d: {list(factors.values())}"
        ),
    }
)
experiments_path.write_text(json.dumps(experiments, indent=2, ensure_ascii=False), encoding="utf-8")
print("Logged to experiments.json")

print("\n" + "=" * 60)
print(f"EXPERIMENT {EXPERIMENT_NAME} COMPLETE")
print(f"  21d partial val (calib): WAPE={w_cal:.4f}  RBias={r_cal:.4f}  Total={t_cal:.4f}")
print("  Test: blended 21d+35d predictions, calibrated from 21d val")
print("=" * 60)
