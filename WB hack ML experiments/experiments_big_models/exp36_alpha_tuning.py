"""Experiment 36: Alpha tuning — find optimal alpha for near-zero raw RBias.

Key insight from exp35:
  - Raw WAPE=0.2388, but RBias=0.0196 (over-predicting by ~2%)
  - Per-step calibration fixes RBias=0 but WAPE rises to 0.2405
  - If we find alpha where sum(pred) ~ sum(actual) naturally,
    calibration has no distortion -> total ~ raw_WAPE ~ 0.2388

Hypothesis: alpha=0.55 over-predicts, alpha=0.52 under-predicts less.
Try 5 seeds x alpha=0.52 + 5 seeds x alpha=0.53 (narrower range than 0.52+0.55).
Monitor raw RBias: if closer to 0, calibrated result improves.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import json
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, ".")

from config import FUTURE_TARGET_COLS
from data import load_data, create_future_targets, build_feature_cols, split_data, encode_categoricals
from features import make_features
from metrics import WapePlusRbias
from train import predict_steps, build_submission

EXPERIMENT_NAME = "exp36_alpha_tuning"
SEEDS = [42, 123, 456, 789, 1234]
ALPHAS = [0.52, 0.53]  # 5 seeds each = 10 models

BASE_LGB = dict(
    n_estimators=5000, learning_rate=0.05, num_leaves=127,
    min_child_samples=20, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    n_jobs=-1, verbose=-1,
)

print("=" * 60)
print(f"EXPERIMENT: {EXPERIMENT_NAME}")
print(f"  10 LGB (5x alpha=0.52 + 5x alpha=0.53) | leaves=127")
print(f"  154 features | per-step calib")
print(f"  Goal: raw RBias closer to 0 -> less calibration distortion")
print("=" * 60)
sys.stdout.flush()


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
        m = lgb.LGBMRegressor(**params)
        m.fit(
            X_fit, y_fit[step_col],
            eval_set=[(X_valid, y_valid[step_col])],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        models[step_col] = m
        best_score = next(iter(m.best_score_.get("valid_0", {}).values()), float("nan"))
        print(f"    {step_col:20s}  iter={m.best_iteration_:4d}  score={best_score:.4f}")
        sys.stdout.flush()
    return models


# -- Data & features ----------------------------------------------------------
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

# -- Train alpha=0.52 ---------------------------------------------------------
print(f"\n--- LGB alpha=0.52 ({len(SEEDS)} seeds) ---")
sys.stdout.flush()
for seed in SEEDS:
    t0 = datetime.now()
    print(f"\n  seed={seed} [{t0.strftime('%H:%M:%S')}]")
    sys.stdout.flush()
    params = {**BASE_LGB, "objective": "quantile", "alpha": 0.52, "random_state": seed}
    models = train_lgb_models(X_fit, y_fit, X_valid, y_valid, params)
    vp = predict_steps(models, X_valid)
    tp = predict_steps(models, X_test)
    w, r, t = metric.calculate_components(y_valid, vp)
    elapsed = (datetime.now() - t0).seconds
    print(f"  seed={seed} a=0.52: WAPE={w:.4f} RBias={r:.4f} Total={t:.4f}  [{elapsed}s]")
    sys.stdout.flush()
    valid_preds_list.append(vp)
    test_preds_list.append(tp)
    joblib.dump(models, models_dir / f"lgb_a052_seed{seed}.pkl")

# -- Train alpha=0.53 ---------------------------------------------------------
print(f"\n--- LGB alpha=0.53 ({len(SEEDS)} seeds) ---")
sys.stdout.flush()
for seed in SEEDS:
    t0 = datetime.now()
    print(f"\n  seed={seed} [{t0.strftime('%H:%M:%S')}]")
    sys.stdout.flush()
    params = {**BASE_LGB, "objective": "quantile", "alpha": 0.53, "random_state": seed}
    models = train_lgb_models(X_fit, y_fit, X_valid, y_valid, params)
    vp = predict_steps(models, X_valid)
    tp = predict_steps(models, X_test)
    w, r, t = metric.calculate_components(y_valid, vp)
    elapsed = (datetime.now() - t0).seconds
    print(f"  seed={seed} a=0.53: WAPE={w:.4f} RBias={r:.4f} Total={t:.4f}  [{elapsed}s]")
    sys.stdout.flush()
    valid_preds_list.append(vp)
    test_preds_list.append(tp)
    joblib.dump(models, models_dir / f"lgb_a053_seed{seed}.pkl")

# -- Ensemble -----------------------------------------------------------------
print(f"\n[Ensemble] Averaging {len(valid_preds_list)} predictions...")
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
print(f"[Raw]        WAPE={w_raw:.4f} RBias={r_raw:.4f} Total={t_raw:.4f}")
sys.stdout.flush()

# Per-step calibration
valid_cal = valid_ens.copy()
test_cal = test_ens.copy()
factors = {}
for c in FUTURE_TARGET_COLS:
    f = float(y_valid[c].sum()) / max(float(valid_ens[c].sum()), 1e-9)
    valid_cal[c] = (valid_ens[c] * f).clip(lower=0)
    test_cal[c] = (test_ens[c] * f).clip(lower=0)
    factors[c] = round(f, 4)

w_cal, r_cal, t_cal = metric.calculate_components(y_valid, valid_cal)
print(f"[Calibrated] WAPE={w_cal:.4f} RBias={r_cal:.4f} Total={t_cal:.4f}")
print(f"[Cal factors per step]: {list(factors.values())}")
sys.stdout.flush()

# -- Submission ---------------------------------------------------------------
sub = build_submission(test_cal, X_test, inference_ts, test_df)
sub_path = f"submission_team_{EXPERIMENT_NAME}.csv"
sub.to_csv(sub_path, index=False)
print(f"Saved: {sub_path}  ({len(sub)} rows)")
sys.stdout.flush()

# -- Log ----------------------------------------------------------------------
exp_path = Path("experiments.json")
experiments = json.loads(exp_path.read_text(encoding="utf-8"))
experiments.append({
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "name": EXPERIMENT_NAME,
    "params": {
        "seeds": SEEDS, "alphas": ALPHAS, "num_leaves": 127,
        "n_models": len(valid_preds_list),
        "features": "extended+rolling_cv+lag_ratios (154)",
        "calibration": "per_step",
    },
    "wape": round(w_cal, 6),
    "rbias": round(r_cal, 6),
    "total": round(t_cal, 6),
    "note": (
        f"5x LGB a=0.52 + 5x LGB a=0.53. "
        f"Raw={t_raw:.4f} (WAPE={w_raw:.4f} RBias={r_raw:.4f}) -> "
        f"Cal={t_cal:.4f}. "
        f"Cal factors: {list(factors.values())}"
    ),
})
exp_path.write_text(json.dumps(experiments, indent=2, ensure_ascii=False), encoding="utf-8")
print("Logged to experiments.json")

print("\n" + "=" * 60)
print(f"EXPERIMENT {EXPERIMENT_NAME} COMPLETE")
print(f"  Raw:         WAPE={w_raw:.4f}  RBias={r_raw:.4f}  Total={t_raw:.4f}")
print(f"  Calibrated:  WAPE={w_cal:.4f}  RBias={r_cal:.4f}  Total={t_cal:.4f}")
print("=" * 60)
