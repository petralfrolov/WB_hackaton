"""Experiment 44: Longer training window — 35 days instead of 21.

Hypothesis:
  val scores plateau at ~0.2405 regardless of capacity/LR tweaks.
  The public test likely contains routes with patterns not captured in 21 days.
  Using 35 days (5 full weeks) gives more seasonal & weekly diversity in training,
  which may improve OOD generalisation on the public test.

Config:
  - TRAIN_DAYS overridden to 35 (vs default 21)
  - num_leaves=511, min_child_samples=10 (best capacity from exp40)
  - alpha=0.55, 10 seeds
  - 154 features (extended + rolling_cv + lag_ratios)
  - per-step calibration
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

# -- override TRAIN_DAYS before any module imports it --
import config as _cfg
_cfg.TRAIN_DAYS = 35

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


EXPERIMENT_NAME = "exp44_window35_leaves511"
SEEDS = [42, 123, 456, 789, 1234, 2024, 7, 314, 99, 888]
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
print(f"  TRAIN_DAYS=35 | num_leaves=511 | alpha={ALPHA} | 10 seeds")
print("  Hypothesis: wider window -> more route diversity -> better OOD")
print("=" * 60)
sys.stdout.flush()

print("Loading data...")
sys.stdout.flush()
train_df, test_df = load_data()
print(f"Routes: {train_df['route_id'].nunique()} | Train rows: {len(train_df)}")
sys.stdout.flush()

print(f"Building features (TRAIN_DAYS override = {_cfg.TRAIN_DAYS})...")
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
    t0 = datetime.now()
    print(f"\n--- LGB win35 leaves=511 alpha={ALPHA} seed={seed} [{t0.strftime('%H:%M:%S')}] ---")
    sys.stdout.flush()
    params = {**BASE_LGB, "objective": "quantile", "alpha": ALPHA, "random_state": seed}
    models = train_lgb_models(X_fit, y_fit, X_valid, y_valid, params)
    vp = predict_steps(models, X_valid)
    tp = predict_steps(models, X_test)
    w, r, t = metric.calculate_components(y_valid, vp)
    elapsed = (datetime.now() - t0).seconds
    print(f"  seed={seed}: WAPE={w:.4f} RBias={r:.4f} Total={t:.4f}  [{elapsed}s]")
    sys.stdout.flush()
    valid_preds_list.append(vp)
    test_preds_list.append(tp)
    joblib.dump(models, models_dir / f"lgb_a055_seed{seed}.pkl")

print(f"\n[Ensemble] Averaging {len(SEEDS)} predictions...")
sys.stdout.flush()

valid_ens = pd.DataFrame(
    np.mean([p.values for p in valid_preds_list], axis=0),
    columns=FUTURE_TARGET_COLS,
    index=X_valid.index,
).clip(lower=0)

test_ens = pd.DataFrame(
    np.mean([p.values for p in test_preds_list], axis=0),
    columns=FUTURE_TARGET_COLS,
    index=X_test.index,
).clip(lower=0)

w_raw, r_raw, t_raw = metric.calculate_components(y_valid, valid_ens)
print(f"[Raw]        WAPE={w_raw:.4f} RBias={r_raw:.4f} Total={t_raw:.4f}")
sys.stdout.flush()

valid_cal = valid_ens.copy()
test_cal = test_ens.copy()
factors = {}
for col in FUTURE_TARGET_COLS:
    factor = float(y_valid[col].sum()) / max(float(valid_ens[col].sum()), 1e-9)
    valid_cal[col] = (valid_ens[col] * factor).clip(lower=0)
    test_cal[col] = (test_ens[col] * factor).clip(lower=0)
    factors[col] = round(factor, 4)

w_cal, r_cal, t_cal = metric.calculate_components(y_valid, valid_cal)
print(f"[Calibrated] WAPE={w_cal:.4f} RBias={r_cal:.4f} Total={t_cal:.4f}")
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
            "train_days": 35,
            "seeds": SEEDS,
            "alpha": ALPHA,
            "num_leaves": 511,
            "min_child_samples": 10,
            "n_models": len(SEEDS),
            "features": "extended+rolling_cv+lag_ratios (154)",
            "calibration": "per_step",
        },
        "wape": round(w_cal, 6),
        "rbias": round(r_cal, 6),
        "total": round(t_cal, 6),
        "note": (
            f"TRAIN_DAYS=35 (vs 21), LGB leaves=511 min_child=10 alpha={ALPHA}. "
            f"Raw: WAPE={w_raw:.4f} RBias={r_raw:.4f}. Cal: total={t_cal:.4f}. "
            f"Factors: {list(factors.values())}"
        ),
    }
)
experiments_path.write_text(json.dumps(experiments, indent=2, ensure_ascii=False), encoding="utf-8")
print("Logged to experiments.json")

print("\n" + "=" * 60)
print(f"EXPERIMENT {EXPERIMENT_NAME} COMPLETE")
print(f"  Raw:         WAPE={w_raw:.4f}  RBias={r_raw:.4f}  Total={t_raw:.4f}")
print(f"  Calibrated:  WAPE={w_cal:.4f}  RBias={r_cal:.4f}  Total={t_cal:.4f}")
print("=" * 60)
