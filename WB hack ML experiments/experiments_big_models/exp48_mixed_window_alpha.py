"""Experiment 48: Mixed window + mixed alpha ensemble.

Combines two sources of diversity that individually gave public LB gains:
  - Window diversity: exp45 (21d+35d)         -> 0.2508  (NEW BEST)
  - The original mixed alpha: exp39/exp43      -> marginal val improvement

Config (10 models total):
  - 4 models: 21-day window, alpha=0.55  [seeds 42,123,456,789]
  - 3 models: 35-day window, alpha=0.55  [seeds 1234,2024,7]
  - 3 models: 21-day window, alpha=0.52  [seeds 314,99,888]

All: leaves=511, min_child=10.
Calibration anchored on 21-day alpha=0.55 val (consistent with exp40/exp47).
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


EXPERIMENT_NAME = "exp48_mixed_window_alpha"
SEEDS_21_A55 = [42, 123, 456, 789]   # 21d, alpha=0.55
SEEDS_35_A55 = [1234, 2024, 7]        # 35d, alpha=0.55
SEEDS_21_A52 = [314, 99, 888]         # 21d, alpha=0.52

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
print(f"  4x(21d,a55) + 3x(35d,a55) + 3x(21d,a52) | leaves=511")
print("  Dual diversity: training window + quantile alpha")
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


def train_group(seeds, window_days, alpha, label, X_fit, y_fit, X_valid, y_valid, X_test, models_dir):
    """Train a group of seeds and return (valid_preds, test_preds)."""
    print(f"\n>>> {label}: {len(seeds)} seeds, window={window_days}d, alpha={alpha} <<<")
    sys.stdout.flush()
    vp_list, tp_list = [], []
    for seed in seeds:
        t0 = datetime.now()
        print(f"\n--- {label} seed={seed} [{t0.strftime('%H:%M:%S')}] ---")
        sys.stdout.flush()
        params = {**BASE_LGB, "objective": "quantile", "alpha": alpha, "random_state": seed}
        models = train_lgb_models(X_fit, y_fit, X_valid, y_valid, params)
        vp = predict_steps(models, X_valid)
        tp = predict_steps(models, X_test)
        w, r, t = metric.calculate_components(y_valid, vp)
        elapsed = (datetime.now() - t0).seconds
        print(f"  seed={seed}: WAPE={w:.4f} RBias={r:.4f} Total={t:.4f}  [{elapsed}s]")
        sys.stdout.flush()
        vp_list.append(vp)
        tp_list.append(tp)
        joblib.dump(models, models_dir / f"lgb_{label}_seed{seed}.pkl")
    return vp_list, tp_list


# ── Load and build features (once) ──────────────────────────────────────────
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

# ── 21-day splits ────────────────────────────────────────────────────────────
print("\nPreparing 21-day split...")
sys.stdout.flush()
_cfg.TRAIN_DAYS = 21
X_fit_21, y_fit_21, X_valid_21, y_valid_21, X_test_21, inference_ts, test_model_df = split_data(
    train_df_feat, feature_cols
)
X_fit_21, X_valid_21, X_test_21, _ = encode_categoricals(
    X_fit_21, X_valid_21, X_test_21, feature_cols
)
print(f"  21d: Fit={len(X_fit_21):,}  Valid={len(X_valid_21):,}")
sys.stdout.flush()

# ── 35-day split ─────────────────────────────────────────────────────────────
print("Preparing 35-day split...")
sys.stdout.flush()
_cfg.TRAIN_DAYS = 35
X_fit_35, y_fit_35, X_valid_35, y_valid_35, X_test_35, _, _ = split_data(
    train_df_feat, feature_cols
)
X_fit_35, X_valid_35, X_test_35, _ = encode_categoricals(
    X_fit_35, X_valid_35, X_test_35, feature_cols
)
print(f"  35d: Fit={len(X_fit_35):,}  Valid={len(X_valid_35):,}")
sys.stdout.flush()

# ── Group 1: 21d, alpha=0.55 ─────────────────────────────────────────────────
vp_21_a55, tp_21_a55 = train_group(
    SEEDS_21_A55, 21, 0.55, "21d_a55",
    X_fit_21, y_fit_21, X_valid_21, y_valid_21, X_test_21, models_dir,
)

# ── Group 2: 35d, alpha=0.55 ─────────────────────────────────────────────────
vp_35_a55, tp_35_a55 = train_group(
    SEEDS_35_A55, 35, 0.55, "35d_a55",
    X_fit_35, y_fit_35, X_valid_35, y_valid_35, X_test_35, models_dir,
)

# ── Group 3: 21d, alpha=0.52 ─────────────────────────────────────────────────
vp_21_a52, tp_21_a52 = train_group(
    SEEDS_21_A52, 21, 0.52, "21d_a52",
    X_fit_21, y_fit_21, X_valid_21, y_valid_21, X_test_21, models_dir,
)

# ── Build test ensemble ──────────────────────────────────────────────────────
all_test_preds = tp_21_a55 + tp_35_a55 + tp_21_a52
n_total = len(all_test_preds)
print(f"\n[Ensemble] Blending {n_total} test predictions...")
sys.stdout.flush()

test_ens = pd.DataFrame(
    np.mean([tp.values for tp in all_test_preds], axis=0),
    columns=FUTURE_TARGET_COLS,
    index=X_test_21.index,
).clip(lower=0)

# ── Validation indicator: 21d alpha=0.55 models (4 seeds, comparable ref) ───
ens_ref = pd.DataFrame(
    np.mean([p.values for p in vp_21_a55], axis=0),
    columns=FUTURE_TARGET_COLS,
    index=X_valid_21.index,
).clip(lower=0)
w_ref_raw, r_ref_raw, t_ref_raw = metric.calculate_components(y_valid_21, ens_ref)
print(f"[21d/a55 4-model raw]   WAPE={w_ref_raw:.4f} RBias={r_ref_raw:.4f} Total={t_ref_raw:.4f}")
sys.stdout.flush()

# ── Calibrate test from 21d/a55 val ─────────────────────────────────────────
factors = {}
for col in FUTURE_TARGET_COLS:
    f = float(y_valid_21[col].sum()) / max(float(ens_ref[col].sum()), 1e-9)
    test_ens[col] = (test_ens[col] * f).clip(lower=0)
    ens_ref[col] = (ens_ref[col] * f).clip(lower=0)
    factors[col] = round(f, 4)
w_ref_cal, r_ref_cal, t_ref_cal = metric.calculate_components(y_valid_21, ens_ref)
print(f"[21d/a55 4-model cal]   WAPE={w_ref_cal:.4f} RBias={r_ref_cal:.4f} Total={t_ref_cal:.4f}")
print(f"[Cal factors]: {list(factors.values())}")
sys.stdout.flush()

# ── Submission ────────────────────────────────────────────────────────────────
sub = build_submission(test_ens, X_test_21, inference_ts, test_df)
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
        "seeds_21d_a55": SEEDS_21_A55,
        "seeds_35d_a55": SEEDS_35_A55,
        "seeds_21d_a52": SEEDS_21_A52,
        "alpha_groups": [0.55, 0.55, 0.52],
        "windows": [21, 35, 21],
        "num_leaves": 511,
        "min_child_samples": 10,
        "n_models": n_total,
        "features": "extended+rolling_cv+lag_ratios (154)",
        "calibration": "per_step from 21d/a55 val",
    },
    "wape": round(w_ref_cal, 6),
    "rbias": round(r_ref_cal, 6),
    "total": round(t_ref_cal, 6),
    "note": (
        f"Mixed window+alpha: 4x21d/a55 + 3x35d/a55 + 3x21d/a52, leaves=511. "
        f"21d/a55 4-model raw val: {t_ref_raw:.4f}. Cal total: {t_ref_cal:.4f}. "
        f"Factors: {list(factors.values())}"
    ),
})
exp_path.write_text(json.dumps(experiments, indent=2, ensure_ascii=False), encoding="utf-8")
print("Logged to experiments.json")

print("\n" + "=" * 60)
print(f"EXPERIMENT {EXPERIMENT_NAME} COMPLETE")
print(f"  21d/a55 val (calib): WAPE={w_ref_cal:.4f}  RBias={r_ref_cal:.4f}  Total={t_ref_cal:.4f}")
print(f"  Test: 10 models (window+alpha diversity), calibrated from 21d/a55 val")
print("=" * 60)
