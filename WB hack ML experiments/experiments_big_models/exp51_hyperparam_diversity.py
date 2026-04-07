"""Experiment 51: Structural hyperparameter diversity — 3 groups x 21d.

Key insight from public LB:
  - True multi-window (42d data) HURTS (exp48-50 worse than exp40/exp45)
  - "Window diversity" in exp45 was actually just seed diversity (broken 35d=21d)
  - Best config remains: 21d window, leaves=511, alpha=0.55
  - exp40/exp45 are both ~0.2508 on public LB

New direction: structural diversity via different colsample_bytree + subsample.
Different column sampling means each model sees different feature subsets per
tree, creating structurally diverse tree ensembles. Averaged predictions should
be smoother and more robust on OOD test data.

Config (10 models, all 21d, alpha=0.55, leaves=511):
  Group A (4 seeds): colsample=0.8, subsample=0.8  (standard)
  Group B (3 seeds): colsample=0.5, subsample=0.7  (aggressive dropout)
  Group C (3 seeds): colsample=0.7, subsample=0.9  (high subsample, low colsample)
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


EXPERIMENT_NAME = "exp51_hyperparam_diversity"
ALPHA = 0.55

GROUPS = [
    # (label, seeds, colsample_bytree, subsample)
    ("A_cs08", [42, 123, 456, 789],  0.8, 0.8),
    ("B_cs05", [1234, 2024, 7],      0.5, 0.7),
    ("C_cs07", [314, 99, 888],       0.7, 0.9),
]

BASE_LGB = dict(
    n_estimators=5000,
    learning_rate=0.05,
    num_leaves=511,
    min_child_samples=10,
    subsample_freq=1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_jobs=-1,
    verbose=-1,
)

print("=" * 60)
print(f"EXPERIMENT: {EXPERIMENT_NAME}")
print("  3 hyperparam groups x 21d | leaves=511 | alpha=0.55")
print("  GroupA: cs=0.8 ss=0.8 | GroupB: cs=0.5 ss=0.7 | GroupC: cs=0.7 ss=0.9")
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

print("Building features...")
sys.stdout.flush()
train_df_feat = make_features(train_df_raw, extended=True)
train_df_feat = add_winning_features(train_df_feat)
train_df_feat = create_future_targets(train_df_feat)
feature_cols = build_feature_cols(train_df_feat)
print(f"Feature count: {len(feature_cols)}")
sys.stdout.flush()

X_fit, y_fit, X_valid, y_valid, X_test, inference_ts, _ = split_data(
    train_df_feat, feature_cols, train_days=21
)
X_fit, X_valid, X_test, _ = encode_categoricals(X_fit, X_valid, X_test, feature_cols)
print(f"21d: Fit={len(X_fit):,}  Valid={len(X_valid):,}  Test={len(X_test):,}")
sys.stdout.flush()

metric = WapePlusRbias()
models_dir = Path("models") / EXPERIMENT_NAME
models_dir.mkdir(parents=True, exist_ok=True)

valid_preds_list = []
test_preds_list = []

# ── Train all groups ──────────────────────────────────────────────────────
for label, seeds, colsample, subsample in GROUPS:
    print(f"\n>>> GROUP {label}: {len(seeds)} seeds | cs={colsample} ss={subsample} <<<")
    sys.stdout.flush()
    for seed in seeds:
        t0 = datetime.now()
        print(f"\n--- {label} seed={seed} [{t0.strftime('%H:%M:%S')}] ---")
        sys.stdout.flush()
        params = {
            **BASE_LGB,
            "objective": "quantile",
            "alpha": ALPHA,
            "random_state": seed,
            "colsample_bytree": colsample,
            "subsample": subsample,
        }
        models = train_lgb_models(X_fit, y_fit, X_valid, y_valid, params)
        vp = predict_steps(models, X_valid)
        tp = predict_steps(models, X_test)
        w, r, t = metric.calculate_components(y_valid, vp)
        elapsed = (datetime.now() - t0).seconds
        print(f"  {label} seed={seed}: WAPE={w:.4f} RBias={r:.4f} Total={t:.4f}  [{elapsed}s]")
        sys.stdout.flush()
        valid_preds_list.append(vp)
        test_preds_list.append(tp)
        joblib.dump(models, models_dir / f"lgb_{label}_seed{seed}.pkl")

# ── Ensemble ──────────────────────────────────────────────────────────────
n_total = len(valid_preds_list)
print(f"\n[Ensemble] Averaging {n_total} predictions...")
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
factors = {}
valid_cal = valid_ens.copy()
test_cal = test_ens.copy()
for col in FUTURE_TARGET_COLS:
    f = float(y_valid[col].sum()) / max(float(valid_ens[col].sum()), 1e-9)
    valid_cal[col] = (valid_ens[col] * f).clip(lower=0)
    test_cal[col] = (test_ens[col] * f).clip(lower=0)
    factors[col] = round(f, 4)
w_cal, r_cal, t_cal = metric.calculate_components(y_valid, valid_cal)
print(f"[Calibrated] WAPE={w_cal:.4f} RBias={r_cal:.4f} Total={t_cal:.4f}")
print(f"[Cal factors]: {list(factors.values())}")
sys.stdout.flush()

# ── Submission ────────────────────────────────────────────────────────────
sub = build_submission(test_cal, X_test, inference_ts, test_df)
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
        "groups": [(g[0], g[1], g[2], g[3]) for g in GROUPS],
        "alpha": ALPHA,
        "num_leaves": 511,
        "min_child_samples": 10,
        "n_models": n_total,
        "train_days": 21,
        "features": "extended+rolling_cv+lag_ratios (154)",
        "calibration": "per_step",
    },
    "wape": round(w_cal, 6),
    "rbias": round(r_cal, 6),
    "total": round(t_cal, 6),
    "note": (
        f"Structural diversity: GroupA(cs=0.8,ss=0.8), GroupB(cs=0.5,ss=0.7), "
        f"GroupC(cs=0.7,ss=0.9). All 21d leaves=511 alpha={ALPHA}. "
        f"Raw: {t_raw:.4f}. Cal: {t_cal:.4f}. Factors: {list(factors.values())}"
    ),
})
exp_path.write_text(json.dumps(experiments, indent=2, ensure_ascii=False), encoding="utf-8")
print("Logged to experiments.json")

print("\n" + "=" * 60)
print(f"EXPERIMENT {EXPERIMENT_NAME} COMPLETE")
print(f"  Raw:         WAPE={w_raw:.4f}  RBias={r_raw:.4f}  Total={t_raw:.4f}")
print(f"  Calibrated:  WAPE={w_cal:.4f}  RBias={r_cal:.4f}  Total={t_cal:.4f}")
print("=" * 60)
