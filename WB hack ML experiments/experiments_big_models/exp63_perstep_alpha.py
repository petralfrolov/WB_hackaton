"""Experiment 63: Per-step quantile alpha tuning.

Hypothesis: forecast steps have different uncertainty horizons.
  - Near steps (1-3): demand pattern cleaner → alpha=0.50 (MAE-like, symmetric)
  - Mid steps (4-7): standard uncertainty → alpha=0.55 (default)
  - Far steps (8-10): longer horizon, more uncertainty → alpha=0.60 (conservative)

Post-calibration should normalize the bias per-step anyway, but the WAPE component
(spread of errors) should benefit from a better-matched quantile level.

Config: same GroupB base (cs=0.5, ss=0.7, leaves=511, 21d, 158 feats).
10 seeds – but now each step model uses a step-specific alpha.
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

EXPERIMENT_NAME = "exp63_perstep_alpha"
SEEDS = [42, 123, 456, 789, 1234, 2024, 7, 314, 99, 888]
COLSAMPLE = 0.5
SUBSAMPLE = 0.7

# Step-specific alpha values
STEP_ALPHAS = {
    "target_step_1":  0.50, "target_step_2":  0.50, "target_step_3":  0.50,
    "target_step_4":  0.55, "target_step_5":  0.55, "target_step_6":  0.55, "target_step_7":  0.55,
    "target_step_8":  0.60, "target_step_9":  0.60, "target_step_10": 0.60,
}

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
print(f"  10 seeds | cs={COLSAMPLE} ss={SUBSAMPLE} | leaves=511 | 21d")
print("  Per-step alpha: steps1-3=0.50, steps4-7=0.55, steps8-10=0.60")
print("=" * 60)
sys.stdout.flush()


def add_winning_features(df):
    for w in [4, 8, 48]:
        sc, mc = f"target_roll_std_{w}", f"target_roll_mean_{w}"
        if sc in df.columns and mc in df.columns:
            df[f"target_cv_{w}"] = df[sc] / (df[mc] + 1e-6)
    if "target_lag_1" in df.columns and "target_lag_48" in df.columns:
        df["lag_ratio_1_48"] = df["target_lag_1"] / (df["target_lag_48"] + 1e-6)
    if "target_lag_1" in df.columns and "target_lag_336" in df.columns:
        df["lag_ratio_1_336"] = df["target_lag_1"] / (df["target_lag_336"] + 1e-6)
    if "target_lag_1" in df.columns and "target_ema_8" in df.columns:
        df["momentum_1_ema8"] = df["target_lag_1"] - df["target_ema_8"]
    if "target_lag_1" in df.columns and "target_ema_24" in df.columns:
        df["momentum_1_ema24"] = df["target_lag_1"] - df["target_ema_24"]
    return df


def add_roll_ratio_features(df):
    if "target_roll_mean_8" in df.columns and "target_roll_mean_96" in df.columns:
        df["roll_ratio_8_96"] = df["target_roll_mean_8"] / (df["target_roll_mean_96"] + 1e-6)
    if "target_roll_mean_16" in df.columns and "target_roll_mean_96" in df.columns:
        df["roll_ratio_16_96"] = df["target_roll_mean_16"] / (df["target_roll_mean_96"] + 1e-6)
    if "target_roll_mean_48" in df.columns and "target_roll_mean_336" in df.columns:
        df["roll_ratio_48_336"] = df["target_roll_mean_48"] / (df["target_roll_mean_336"] + 1e-6)
    if "target_ema_8" in df.columns and "target_ema_96" in df.columns:
        df["ema_ratio_8_96"] = df["target_ema_8"] / (df["target_ema_96"] + 1e-6)
    return df


def train_perstep_alpha_models(X_fit, y_fit, X_valid, y_valid, base_params, step_alphas):
    """Train one model per step, each with its own alpha."""
    models = {}
    for step_col in FUTURE_TARGET_COLS:
        alpha = step_alphas[step_col]
        params = {**base_params, "objective": "quantile", "alpha": alpha}
        m = lgb.LGBMRegressor(**params)
        m.fit(
            X_fit, y_fit[step_col],
            eval_set=[(X_valid, y_valid[step_col])],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        models[step_col] = m
        best = next(iter(m.best_score_.get("valid_0", {}).values()), float("nan"))
        print(f"    {step_col:20s}  alpha={alpha:.2f}  iter={m.best_iteration_:4d}  score={best:.4f}")
        sys.stdout.flush()
    return models


print("Loading + building features...")
sys.stdout.flush()
train_df_raw, test_df = load_data()
train_df = make_features(train_df_raw, extended=True)
train_df = add_winning_features(train_df)
train_df = add_roll_ratio_features(train_df)
train_df = create_future_targets(train_df)
feature_cols = build_feature_cols(train_df)
print(f"Feature count: {len(feature_cols)}")
sys.stdout.flush()

X_fit, y_fit, X_valid, y_valid, X_test, inference_ts, _ = split_data(
    train_df, feature_cols, train_days=21
)
X_fit, X_valid, X_test, _ = encode_categoricals(X_fit, X_valid, X_test, feature_cols)
print(f"Fit={len(X_fit):,}  Valid={len(X_valid):,}")
sys.stdout.flush()

metric = WapePlusRbias()
models_dir = Path("models") / EXPERIMENT_NAME
models_dir.mkdir(parents=True, exist_ok=True)
vp_list, tp_list = [], []

base_params = {
    **BASE_LGB,
    "random_state": None,   # will be set per seed
    "colsample_bytree": COLSAMPLE,
    "subsample": SUBSAMPLE,
}

for seed in SEEDS:
    t0 = datetime.now()
    print(f"\n--- per-step-alpha seed={seed} [{t0.strftime('%H:%M:%S')}] ---")
    sys.stdout.flush()
    params = {**base_params, "random_state": seed}
    models = train_perstep_alpha_models(X_fit, y_fit, X_valid, y_valid, params, STEP_ALPHAS)
    vp = predict_steps(models, X_valid)
    tp = predict_steps(models, X_test)
    w, r, t = metric.calculate_components(y_valid, vp)
    print(f"  seed={seed}: WAPE={w:.4f} RBias={r:.4f} Total={t:.4f}  [{(datetime.now()-t0).seconds}s]")
    sys.stdout.flush()
    vp_list.append(vp)
    tp_list.append(tp)
    joblib.dump(models, models_dir / f"lgb_seed{seed}.pkl")


print(f"\n[Ensemble] Averaging {len(SEEDS)} predictions...")
sys.stdout.flush()
valid_ens = pd.DataFrame(
    np.mean([p.values for p in vp_list], axis=0),
    columns=FUTURE_TARGET_COLS, index=X_valid.index,
).clip(lower=0)
test_ens = pd.DataFrame(
    np.mean([p.values for p in tp_list], axis=0),
    columns=FUTURE_TARGET_COLS, index=X_test.index,
).clip(lower=0)

w_raw, r_raw, t_raw = metric.calculate_components(y_valid, valid_ens)
print(f"[Raw]        WAPE={w_raw:.4f} RBias={r_raw:.4f} Total={t_raw:.4f}")

factors = {}
valid_cal, test_cal = valid_ens.copy(), test_ens.copy()
for col in FUTURE_TARGET_COLS:
    f = float(y_valid[col].sum()) / max(float(valid_ens[col].sum()), 1e-9)
    valid_cal[col] = (valid_ens[col] * f).clip(lower=0)
    test_cal[col]  = (test_ens[col]  * f).clip(lower=0)
    factors[col] = round(f, 4)

w_cal, r_cal, t_cal = metric.calculate_components(y_valid, valid_cal)
print(f"[Calibrated] WAPE={w_cal:.4f} RBias={r_cal:.4f} Total={t_cal:.4f}")
print(f"[Cal factors]: {list(factors.values())}")
sys.stdout.flush()

sub = build_submission(test_cal, X_test, inference_ts, test_df)
sub.to_csv(f"submission_team_{EXPERIMENT_NAME}.csv", index=False)
print(f"Saved: submission_team_{EXPERIMENT_NAME}.csv  ({len(sub)} rows)")

exp_path = Path("experiments.json")
exps = json.loads(exp_path.read_text(encoding="utf-8"))
exps.append({
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "name": EXPERIMENT_NAME,
    "params": {
        "seeds": SEEDS,
        "step_alphas": STEP_ALPHAS,
        "num_leaves": 511,
        "colsample_bytree": COLSAMPLE,
        "subsample": SUBSAMPLE,
        "n_models": len(SEEDS),
        "train_days": 21,
        "features": "158 (154+4 roll-ratio)",
        "calibration": "per_step",
    },
    "wape": round(w_cal, 6),
    "rbias": round(r_cal, 6),
    "total": round(t_cal, 6),
    "note": (
        f"Per-step alpha: steps1-3=0.50, steps4-7=0.55, steps8-10=0.60. "
        f"cs=0.5 ss=0.7 leaves=511 10 seeds. "
        f"Raw: WAPE={w_raw:.4f} RBias={r_raw:.4f} Total={t_raw:.4f}. "
        f"Cal: total={t_cal:.4f}. Factors: {list(factors.values())}"
    ),
})
exp_path.write_text(json.dumps(exps, indent=2, ensure_ascii=False), encoding="utf-8")

print("\n" + "=" * 60)
print(f"EXPERIMENT {EXPERIMENT_NAME} COMPLETE")
print(f"  Per-step alpha: near=0.50 / mid=0.55 / far=0.60")
print(f"  Raw:        WAPE={w_raw:.4f}  RBias={r_raw:.4f}  Total={t_raw:.4f}")
print(f"  Calibrated: WAPE={w_cal:.4f}  RBias={r_cal:.4f}  Total={t_cal:.4f}")
print(f"  vs exp53 baseline: 0.239683")
print("=" * 60)
