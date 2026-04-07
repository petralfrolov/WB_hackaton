"""Experiment 47: Triple-window ensemble — 4x21d + 3x35d + 3x49d.

Motivation:
  exp40  (21d, leaves=511)          -> public 0.251
  exp45  (21d+35d mix, leaves=511)  -> public 0.2508  *** NEW BEST ***

  Window diversity helps. Extending to THREE windows maximises temporal spread
  while staying at 10 models.

Config:
  - 21-day window: seeds [42, 123, 456, 789]          (4 models)
  - 35-day window: seeds [1234, 2024, 7]               (3 models)
  - 49-day window: seeds [314, 99, 888]                (3 models)
  - leaves=511, min_child_samples=10, alpha=0.55
  - Calibration anchored on 21-day validation split (consistent reference)
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


EXPERIMENT_NAME = "exp47_triple_window"
SEEDS_21 = [42, 123, 456, 789]
SEEDS_35 = [1234, 2024, 7]
SEEDS_49 = [314, 99, 888]
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

print("=" * 60)
print(f"EXPERIMENT: {EXPERIMENT_NAME}")
print(f"  4x21d + 3x35d + 3x49d | leaves=511 | alpha={ALPHA}")
print("  Max temporal diversity, 10 models total")
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


def run_window(seeds, window_days, label, train_df_feat, feature_cols, models_dir):
    """Train models for one window size, return (valid_preds_list, test_preds_list)."""
    print(f"\n>>> WINDOW = {window_days} days <<<")
    sys.stdout.flush()
    _cfg.TRAIN_DAYS = window_days
    X_fit, y_fit, X_valid, y_valid, X_test, _, _ = split_data(train_df_feat, feature_cols)
    X_fit, X_valid, X_test, _ = encode_categoricals(X_fit, X_valid, X_test, feature_cols)
    print(f"  Fit={len(X_fit):,}  Valid={len(X_valid):,}  Test={len(X_test):,}")
    sys.stdout.flush()

    vp_list, tp_list = [], []
    for seed in seeds:
        t0 = datetime.now()
        print(f"\n--- {window_days}d leaves=511 alpha={ALPHA} seed={seed} [{t0.strftime('%H:%M:%S')}] ---")
        sys.stdout.flush()
        params = {**BASE_LGB, "objective": "quantile", "alpha": ALPHA, "random_state": seed}
        models = train_lgb_models(X_fit, y_fit, X_valid, y_valid, params)
        vp = predict_steps(models, X_valid)
        tp = predict_steps(models, X_test)
        w, r, t = metric.calculate_components(y_valid, vp)
        elapsed = (datetime.now() - t0).seconds
        print(f"  seed={seed} {window_days}d: WAPE={w:.4f} RBias={r:.4f} Total={t:.4f}  [{elapsed}s]")
        sys.stdout.flush()
        vp_list.append(vp)
        tp_list.append(tp)
        joblib.dump(models, models_dir / f"lgb_{window_days}d_seed{seed}.pkl")
    return vp_list, tp_list


# ── Load & prepare features (once) ──────────────────────────────────────────
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

# 21-day reference split (for calibration anchor + submission build)
_cfg.TRAIN_DAYS = 21
X_fit_ref, y_fit_ref, X_valid_ref, y_valid_ref, X_test_ref, inference_ts, test_model_df = split_data(
    train_df_feat, feature_cols
)
X_fit_ref, X_valid_ref, X_test_ref, _ = encode_categoricals(
    X_fit_ref, X_valid_ref, X_test_ref, feature_cols
)

metric = WapePlusRbias()
models_dir = Path("models") / EXPERIMENT_NAME
models_dir.mkdir(parents=True, exist_ok=True)

# ── Train all three windows ──────────────────────────────────────────────────
vp_21, tp_21 = run_window(SEEDS_21, 21, "21d", train_df_feat, feature_cols, models_dir)
vp_35, tp_35 = run_window(SEEDS_35, 35, "35d", train_df_feat, feature_cols, models_dir)
vp_49, tp_49 = run_window(SEEDS_49, 49, "49d", train_df_feat, feature_cols, models_dir)

# ── Build test ensemble ──────────────────────────────────────────────────────
# All test predictions are on the same 1000-route test set, so we average directly.
all_test_preds = tp_21 + tp_35 + tp_49
test_ens = pd.DataFrame(
    np.mean([tp.values for tp in all_test_preds], axis=0),
    columns=FUTURE_TARGET_COLS,
    index=X_test_ref.index,
).clip(lower=0)

# ── Validation indicator (21-day models only — comparable reference) ─────────
print(f"\n[Ensemble] 10 test models blended ({len(SEEDS_21)} x 21d + {len(SEEDS_35)} x 35d + {len(SEEDS_49)} x 49d)")
sys.stdout.flush()

ens_21_valid = pd.DataFrame(
    np.mean([p.values for p in vp_21], axis=0),
    columns=FUTURE_TARGET_COLS,
    index=X_valid_ref.index,
).clip(lower=0)
w_21_raw, r_21_raw, t_21_raw = metric.calculate_components(y_valid_ref, ens_21_valid)
print(f"[21d 4-model raw]   WAPE={w_21_raw:.4f} RBias={r_21_raw:.4f} Total={t_21_raw:.4f}")
sys.stdout.flush()

# ── Calibrate test from 21d val ───────────────────────────────────────────────
factors = {}
for col in FUTURE_TARGET_COLS:
    f = float(y_valid_ref[col].sum()) / max(float(ens_21_valid[col].sum()), 1e-9)
    test_ens[col] = (test_ens[col] * f).clip(lower=0)
    factors[col] = round(f, 4)

ens_21_cal = ens_21_valid.copy()
for col in FUTURE_TARGET_COLS:
    ens_21_cal[col] = (ens_21_valid[col] * factors[col]).clip(lower=0)
w_21_cal, r_21_cal, t_21_cal = metric.calculate_components(y_valid_ref, ens_21_cal)
print(f"[21d 4-model cal]   WAPE={w_21_cal:.4f} RBias={r_21_cal:.4f} Total={t_21_cal:.4f}")
print(f"[Cal factors]: {list(factors.values())}")
sys.stdout.flush()

# ── Submission ────────────────────────────────────────────────────────────────
sub = build_submission(test_ens, X_test_ref, inference_ts, test_df)
sub_path = f"submission_team_{EXPERIMENT_NAME}.csv"
sub.to_csv(sub_path, index=False)
print(f"Saved: {sub_path}  ({len(sub)} rows)")
sys.stdout.flush()

# ── Log ───────────────────────────────────────────────────────────────────────
exp_path = Path("experiments.json")
experiments = json.loads(exp_path.read_text(encoding="utf-8"))
experiments.append({
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "name": EXPERIMENT_NAME,
    "params": {
        "seeds_21d": SEEDS_21,
        "seeds_35d": SEEDS_35,
        "seeds_49d": SEEDS_49,
        "alpha": ALPHA,
        "num_leaves": 511,
        "min_child_samples": 10,
        "n_models": len(SEEDS_21) + len(SEEDS_35) + len(SEEDS_49),
        "features": "extended+rolling_cv+lag_ratios (154)",
        "calibration": "per_step from 21d val",
    },
    "wape": round(w_21_cal, 6),
    "rbias": round(r_21_cal, 6),
    "total": round(t_21_cal, 6),
    "note": (
        f"Triple window {len(SEEDS_21)}x21d + {len(SEEDS_35)}x35d + {len(SEEDS_49)}x49d, "
        f"leaves=511 alpha={ALPHA}. "
        f"21d val raw: WAPE={w_21_raw:.4f} RBias={r_21_raw:.4f}. "
        f"21d val cal: total={t_21_cal:.4f}. Factors: {list(factors.values())}"
    ),
})
exp_path.write_text(json.dumps(experiments, indent=2, ensure_ascii=False), encoding="utf-8")
print("Logged to experiments.json")

print("\n" + "=" * 60)
print(f"EXPERIMENT {EXPERIMENT_NAME} COMPLETE")
print(f"  21d val (calib): WAPE={w_21_cal:.4f}  RBias={r_21_cal:.4f}  Total={t_21_cal:.4f}")
print(f"  Test: blended {len(SEEDS_21)+len(SEEDS_35)+len(SEEDS_49)} models (3 windows), calibrated from 21d val")
print("=" * 60)
