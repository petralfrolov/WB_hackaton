"""Experiment 46: leaves=511 with original min_child_samples=20.

Capacity sweet spot is clearly leaves=511 (public 0.251).
Trying tighter leaf regularisation: min_child=20 (baseline value) instead of 10.

Reasoning:
  - exp40 (511, min_child=10)  -> 0.251 public (best)
  - exp41 (1023, min_child=5)  -> 0.2538 (overfit regression)
  Looser leaf floor (5) caused overfit at 1023. Tighter floor (20) at 511 may
  produce a MORE regularised version that avoids val noise without hitting the
  over-capacity regime.

Config:
  - 10 seeds x alpha=0.55
  - num_leaves=511, min_child_samples=20
  - learning_rate=0.05, n_estimators=5000, early_stopping=100
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


EXPERIMENT_NAME = "exp46_leaves511_minchild20"
SEEDS = [42, 123, 456, 789, 1234, 2024, 7, 314, 99, 888]
ALPHA = 0.55

BASE_LGB = dict(
    n_estimators=5000,
    learning_rate=0.05,
    num_leaves=511,
    min_child_samples=20,
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
print(f"  10 LGB | num_leaves=511 | min_child=20 | alpha={ALPHA}")
print("  Tighter leaf regularisation vs exp40 (min_child=10)")
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
    t0 = datetime.now()
    print(f"\n--- LGB leaves=511 mc20 alpha={ALPHA} seed={seed} [{t0.strftime('%H:%M:%S')}] ---")
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
            "seeds": SEEDS,
            "alpha": ALPHA,
            "num_leaves": 511,
            "min_child_samples": 20,
            "n_models": len(SEEDS),
            "features": "extended+rolling_cv+lag_ratios (154)",
            "calibration": "per_step",
        },
        "wape": round(w_cal, 6),
        "rbias": round(r_cal, 6),
        "total": round(t_cal, 6),
        "note": (
            f"LGB leaves=511 min_child=20 alpha={ALPHA}, 10 seeds. "
            f"Raw: WAPE={w_raw:.4f} RBias={r_raw:.4f}. Cal: total={t_cal:.4f}. "
            f"Cal factors: {list(factors.values())}"
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
