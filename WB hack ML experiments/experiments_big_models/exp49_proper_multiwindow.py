"""Experiment 49: PROPER multi-window ensemble using train_days parameter.

Fix: data.py split_data() now accepts train_days kwarg, so we can correctly
train models on different historical windows in a single script.

Public LB context:
  exp40  (10x21d, a55, leaves=511)         -> 0.2510
  exp45  (5x21d + 5x"35d", a55, leaves=511) -> 0.2508  (35d was broken = 21d!)

Since exp45's "window diversity" was actually just seed diversity, it's still
unclear whether TRUE window diversity helps. This experiment finally tests it.

Config (10 models):
  - 5 models: 21-day window, alpha=0.55  [seeds 42,123,456,789,1234]
  - 5 models: 42-day window, alpha=0.55  [seeds 2024,7,314,99,888]
  All: leaves=511, min_child=10.
  Calibration from 21d val (consistent reference).
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


EXPERIMENT_NAME = "exp49_proper_multiwindow"
SEEDS_21 = [42, 123, 456, 789, 1234]
SEEDS_42 = [2024, 7, 314, 99, 888]
ALPHA = 0.55
WINDOW_A = 21
WINDOW_B = 42

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

print("=" * 60)
print(f"EXPERIMENT: {EXPERIMENT_NAME}")
print(f"  5x21d + 5x42d | leaves=511 | alpha={ALPHA}")
print("  TRUE window diversity via split_data(train_days=N)")
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


# ── Load & build features ──────────────────────────────────────────────────
print("Loading data...")
sys.stdout.flush()
train_df_raw, test_df = load_data()
print(f"Routes: {train_df_raw['route_id'].nunique()} | Rows: {len(train_df_raw):,}")
sys.stdout.flush()

print("Building features...")
sys.stdout.flush()
train_df_feat = make_features(train_df_raw, extended=True)
train_df_feat = add_winning_features(train_df_feat)
train_df_feat = create_future_targets(train_df_feat)
feature_cols = build_feature_cols(train_df_feat)
print(f"Feature count: {len(feature_cols)}")
sys.stdout.flush()

metric = WapePlusRbias()
models_dir = Path("models") / EXPERIMENT_NAME
models_dir.mkdir(parents=True, exist_ok=True)

# ── 21-day split (calibration reference) ──────────────────────────────────
print(f"\nPreparing {WINDOW_A}-day split (reference)...")
sys.stdout.flush()
X_fit_21, y_fit_21, X_valid_21, y_valid_21, X_test_21, inference_ts, _ = split_data(
    train_df_feat, feature_cols, train_days=WINDOW_A
)
X_fit_21, X_valid_21, X_test_21, _ = encode_categoricals(
    X_fit_21, X_valid_21, X_test_21, feature_cols
)
print(f"  {WINDOW_A}d: Fit={len(X_fit_21):,}  Valid={len(X_valid_21):,}  Test={len(X_test_21):,}")
sys.stdout.flush()

# ── 42-day split ───────────────────────────────────────────────────────────
print(f"\nPreparing {WINDOW_B}-day split...")
sys.stdout.flush()
X_fit_42, y_fit_42, X_valid_42, y_valid_42, X_test_42, _, _ = split_data(
    train_df_feat, feature_cols, train_days=WINDOW_B
)
X_fit_42, X_valid_42, X_test_42, _ = encode_categoricals(
    X_fit_42, X_valid_42, X_test_42, feature_cols
)
print(f"  {WINDOW_B}d: Fit={len(X_fit_42):,}  Valid={len(X_valid_42):,}  Test={len(X_test_42):,}")
sys.stdout.flush()

# ── Group 1: 21d alpha=0.55 ────────────────────────────────────────────────
print(f"\n>>> GROUP 1: {len(SEEDS_21)} seeds x 21d x alpha={ALPHA} <<<")
sys.stdout.flush()
valid_preds_21, test_preds_21 = [], []
for seed in SEEDS_21:
    t0 = datetime.now()
    print(f"\n--- 21d a={ALPHA} seed={seed} [{t0.strftime('%H:%M:%S')}] ---")
    sys.stdout.flush()
    params = {**BASE_LGB, "objective": "quantile", "alpha": ALPHA, "random_state": seed}
    models = train_lgb_models(X_fit_21, y_fit_21, X_valid_21, y_valid_21, params)
    vp = predict_steps(models, X_valid_21)
    tp = predict_steps(models, X_test_21)
    w, r, t = metric.calculate_components(y_valid_21, vp)
    elapsed = (datetime.now() - t0).seconds
    print(f"  seed={seed}: WAPE={w:.4f} RBias={r:.4f} Total={t:.4f}  [{elapsed}s]")
    sys.stdout.flush()
    valid_preds_21.append(vp)
    test_preds_21.append(tp)
    joblib.dump(models, models_dir / f"lgb_21d_seed{seed}.pkl")

# Partial 21d ensemble
ens_21 = pd.DataFrame(
    np.mean([p.values for p in valid_preds_21], axis=0),
    columns=FUTURE_TARGET_COLS, index=X_valid_21.index,
).clip(lower=0)
w_21, r_21, t_21 = metric.calculate_components(y_valid_21, ens_21)
print(f"\n[21d 5-seed raw] WAPE={w_21:.4f} RBias={r_21:.4f} Total={t_21:.4f}")
sys.stdout.flush()

# ── Group 2: 42d alpha=0.55 ────────────────────────────────────────────────
print(f"\n>>> GROUP 2: {len(SEEDS_42)} seeds x 42d x alpha={ALPHA} <<<")
sys.stdout.flush()
valid_preds_42, test_preds_42 = [], []
for seed in SEEDS_42:
    t0 = datetime.now()
    print(f"\n--- 42d a={ALPHA} seed={seed} [{t0.strftime('%H:%M:%S')}] ---")
    sys.stdout.flush()
    params = {**BASE_LGB, "objective": "quantile", "alpha": ALPHA, "random_state": seed}
    models = train_lgb_models(X_fit_42, y_fit_42, X_valid_42, y_valid_42, params)
    vp_42 = predict_steps(models, X_valid_42)
    tp = predict_steps(models, X_test_42)
    w, r, t = metric.calculate_components(y_valid_42, vp_42)
    elapsed = (datetime.now() - t0).seconds
    print(f"  seed={seed}: WAPE={w:.4f} RBias={r:.4f} Total={t:.4f}  [{elapsed}s]")
    sys.stdout.flush()
    # store test pred; for val metric we use 21d as reference, but store 42d val for info
    valid_preds_42.append(vp_42)
    test_preds_42.append(tp)
    joblib.dump(models, models_dir / f"lgb_42d_seed{seed}.pkl")

# ── Blend test predictions ─────────────────────────────────────────────────
all_test_preds = test_preds_21 + test_preds_42
n_total = len(all_test_preds)
print(f"\n[Ensemble] Blending {n_total} test predictions ({len(SEEDS_21)}x21d + {len(SEEDS_42)}x42d)...")
sys.stdout.flush()

test_ens = pd.DataFrame(
    np.mean([tp.values for tp in all_test_preds], axis=0),
    columns=FUTURE_TARGET_COLS,
    index=X_test_21.index,
).clip(lower=0)

# ── Calibration from 21d val ───────────────────────────────────────────────
factors = {}
ens_21_cal = ens_21.copy()
for col in FUTURE_TARGET_COLS:
    f = float(y_valid_21[col].sum()) / max(float(ens_21[col].sum()), 1e-9)
    ens_21_cal[col] = (ens_21[col] * f).clip(lower=0)
    test_ens[col] = (test_ens[col] * f).clip(lower=0)
    factors[col] = round(f, 4)
w_cal, r_cal, t_cal = metric.calculate_components(y_valid_21, ens_21_cal)
print(f"[21d 5-model cal]  WAPE={w_cal:.4f} RBias={r_cal:.4f} Total={t_cal:.4f}")
print(f"[Cal factors]: {list(factors.values())}")
sys.stdout.flush()

# ── Submission ────────────────────────────────────────────────────────────
sub = build_submission(test_ens, X_test_21, inference_ts, test_df)
sub_path = f"submission_team_{EXPERIMENT_NAME}.csv"
sub.to_csv(sub_path, index=False)
print(f"Saved: {sub_path}  ({len(sub)} rows)")
sys.stdout.flush()

# ── Log ───────────────────────────────────────────────────────────────────
exp_path = Path("experiments.json")
experiments = json.loads(exp_path.read_text(encoding="utf-8"))
experiments.append({
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "name": EXPERIMENT_NAME,
    "params": {
        "seeds_21d": SEEDS_21,
        "seeds_42d": SEEDS_42,
        "alpha": ALPHA,
        "windows": [WINDOW_A, WINDOW_B],
        "num_leaves": 511,
        "min_child_samples": 10,
        "n_models": n_total,
        "features": "extended+rolling_cv+lag_ratios (154)",
        "calibration": "per_step from 21d val",
        "note_fix": "uses split_data(train_days=N) - TRUE window diversity",
    },
    "wape": round(w_cal, 6),
    "rbias": round(r_cal, 6),
    "total": round(t_cal, 6),
    "note": (
        f"PROPER multi-window: 5x21d + 5x42d, leaves=511 alpha={ALPHA}. "
        f"21d 5-model raw: {t_21:.4f}. Cal total: {t_cal:.4f}. "
        f"Factors: {list(factors.values())}"
    ),
})
exp_path.write_text(json.dumps(experiments, indent=2, ensure_ascii=False), encoding="utf-8")
print("Logged to experiments.json")

print("\n" + "=" * 60)
print(f"EXPERIMENT {EXPERIMENT_NAME} COMPLETE")
print(f"  21d val (calib):  WAPE={w_cal:.4f}  RBias={r_cal:.4f}  Total={t_cal:.4f}")
print(f"  Test: {n_total} models (5x{WINDOW_A}d + 5x{WINDOW_B}d), calibrated from 21d val")
print("=" * 60)
