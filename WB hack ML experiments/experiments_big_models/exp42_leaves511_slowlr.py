"""Experiment 42: High-capacity LGB with slower learning.

Public trend:
  leaves=127  -> 0.2530
  leaves=255  -> 0.2515
  leaves=511  -> 0.2510
  leaves=1023 -> 0.2538 (overfit)

Hypothesis:
  leaves=511 looks like the useful capacity zone.
  Instead of increasing leaves further, keep 511 and make boosting smoother:
  lower learning_rate with more estimators may preserve the public gain while
  reducing the overfitting pressure of aggressive tree updates.

Config:
  - 10 seeds x alpha=0.55
  - num_leaves=511, min_child_samples=10
  - learning_rate=0.03, n_estimators=8000
  - 154 features (extended + rolling_cv + lag_ratios)
  - per-step bias calibration
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


EXPERIMENT_NAME = "exp42_leaves511_slowlr"
SEEDS = [42, 123, 456, 789, 1234, 2024, 7, 314, 99, 888]
ALPHA = 0.55

BASE_LGB = dict(
    n_estimators=8000,
    learning_rate=0.03,
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
                lgb.early_stopping(stopping_rounds=150, verbose=False),
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
print(f"  10 LGB seeds | num_leaves=511 | min_child=10 | alpha={ALPHA}")
print("  learning_rate=0.03 | n_estimators=8000 | per-step calib")
print("  Hypothesis: smoother boosting at the exp40 capacity sweet spot")
print("=" * 60)
sys.stdout.flush()

print("Loading data...")
sys.stdout.flush()
train_df, test_df = load_data()
print(f"Routes: {train_df['route_id'].nunique()} | Train rows: {len(train_df)}")
sys.stdout.flush()

print("Building features...")
sys.stdout.flush()
train_df = make_features(train_df, extended=True)
train_df = add_winning_features(train_df)
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

for seed in SEEDS:
    started_at = datetime.now()
    print(f"\n--- LGB leaves=511 slowlr alpha={ALPHA} seed={seed} [{started_at.strftime('%H:%M:%S')}] ---")
    sys.stdout.flush()
    params = {**BASE_LGB, "objective": "quantile", "alpha": ALPHA, "random_state": seed}
    models = train_lgb_models(X_fit, y_fit, X_valid, y_valid, params)
    valid_pred = predict_steps(models, X_valid)
    test_pred = predict_steps(models, X_test)
    wape, rbias, total = metric.calculate_components(y_valid, valid_pred)
    elapsed = (datetime.now() - started_at).seconds
    print(f"  seed={seed}: WAPE={wape:.4f} RBias={rbias:.4f} Total={total:.4f}  [{elapsed}s]")
    sys.stdout.flush()
    valid_preds_list.append(valid_pred)
    test_preds_list.append(test_pred)
    joblib.dump(models, models_dir / f"lgb_a055_seed{seed}.pkl")

print(f"\n[Ensemble] Averaging {len(SEEDS)} predictions...")
sys.stdout.flush()

valid_ens = pd.DataFrame(
    np.mean([pred.values for pred in valid_preds_list], axis=0),
    columns=FUTURE_TARGET_COLS,
    index=X_valid.index,
).clip(lower=0)

test_ens = pd.DataFrame(
    np.mean([pred.values for pred in test_preds_list], axis=0),
    columns=FUTURE_TARGET_COLS,
    index=X_test.index,
).clip(lower=0)

wape_raw, rbias_raw, total_raw = metric.calculate_components(y_valid, valid_ens)
print(f"[Raw]        WAPE={wape_raw:.4f} RBias={rbias_raw:.4f} Total={total_raw:.4f}")
sys.stdout.flush()

valid_cal = valid_ens.copy()
test_cal = test_ens.copy()
factors = {}
for col in FUTURE_TARGET_COLS:
    factor = float(y_valid[col].sum()) / max(float(valid_ens[col].sum()), 1e-9)
    valid_cal[col] = (valid_ens[col] * factor).clip(lower=0)
    test_cal[col] = (test_ens[col] * factor).clip(lower=0)
    factors[col] = round(factor, 4)

wape_cal, rbias_cal, total_cal = metric.calculate_components(y_valid, valid_cal)
print(f"[Calibrated] WAPE={wape_cal:.4f} RBias={rbias_cal:.4f} Total={total_cal:.4f}")
print(f"[Cal factors]: {list(factors.values())}")
sys.stdout.flush()

submission = build_submission(test_cal, X_test, inference_ts, test_df)
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
            "seeds": SEEDS,
            "alpha": ALPHA,
            "num_leaves": 511,
            "min_child_samples": 10,
            "learning_rate": 0.03,
            "n_estimators": 8000,
            "n_models": len(SEEDS),
            "features": "extended+rolling_cv+lag_ratios (154)",
            "calibration": "per_step",
        },
        "wape": round(wape_cal, 6),
        "rbias": round(rbias_cal, 6),
        "total": round(total_cal, 6),
        "note": (
            f"LGB leaves=511 min_child=10 slowlr, 10 seeds alpha={ALPHA}. "
            f"Raw: WAPE={wape_raw:.4f} RBias={rbias_raw:.4f}. Cal: total={total_cal:.4f}. "
            f"Factors: {list(factors.values())}"
        ),
    }
)
experiments_path.write_text(json.dumps(experiments, indent=2, ensure_ascii=False), encoding="utf-8")
print("Logged to experiments.json")

print("\n" + "=" * 60)
print(f"EXPERIMENT {EXPERIMENT_NAME} COMPLETE")
print(f"  Raw:         WAPE={wape_raw:.4f}  RBias={rbias_raw:.4f}  Total={total_raw:.4f}")
print(f"  Calibrated:  WAPE={wape_cal:.4f}  RBias={rbias_cal:.4f}  Total={total_cal:.4f}")
print("=" * 60)