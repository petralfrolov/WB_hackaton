"""Experiment 62: feature_fraction_bynode — per-node feature resampling.

Current state (exp61 result TBD): roll-ratio features test.
baseline: exp57=0.239609, exp53=0.239683.

Hypothesis: feature_fraction_bynode (per-NODE feature sampling) creates more
Random-Forest-like tree diversity compared to colsample_bytree (per-TREE).
In RF, each split randomly considers sqrt(n_features) — this creates heavily
decorrelated trees. LightGBM's feature_fraction_bynode does the same.

Key difference from existing experiments:
  - colsample_bytree=0.5: each tree uses random 50% of features (fixed per tree)
  - feature_fraction_bynode=0.5: each NODE uses random 50% of features (varies per split)
  → bynode is more aggressive diversity: each node in a tree sees different features
  → could improve generalization by preventing overfitting to specific feature subsets

Config (10 models):
  Group A (5 seeds): cs=0.5, ss=0.7, feature_fraction_bynode=0.7  ← standard cs + node-diversity
  Group B (5 seeds): cs=1.0, ss=0.8, feature_fraction_bynode=0.5  ← pure node-diversity (like RF)
  Both: leaves=511, min_child=10, alpha=0.55, 21d, 158 features (if exp61 added new ones)

Expectation: If bynode diversity helps, Group B (RF-style) should have worse individual
 but better ensemble score. Combined ensemble might outperform pure cs=0.5 approach.
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

EXPERIMENT_NAME = "exp62_bynode_diversity"
ALPHA = 0.55

GROUPS = [
    # (label, seeds, colsample_bytree, subsample, feature_fraction_bynode)
    ("A_cs05_bynode07", [42, 123, 456, 789, 1234],    0.5, 0.7, 0.7),
    ("B_cs10_bynode05", [2024, 7, 314, 99, 888],      1.0, 0.8, 0.5),
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
print("  Group A: cs=0.5 ss=0.7 bynode=0.7 (5 seeds)")
print("  Group B: cs=1.0 ss=0.8 bynode=0.5 (5 seeds) [RF-style]")
print(f"  leaves=511 | alpha={ALPHA} | 21d")
print("=" * 60)
sys.stdout.flush()


def add_winning_features(df):
    """Proven features from wave-1/2 screening."""
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


def add_roll_ratio_features(df):
    """New roll-ratio features (4h/2d, 8h/2d, 24h/7d, EMA short/med)."""
    if "target_roll_mean_8" in df.columns and "target_roll_mean_96" in df.columns:
        df["roll_ratio_8_96"] = df["target_roll_mean_8"] / (df["target_roll_mean_96"] + 1e-6)
    if "target_roll_mean_16" in df.columns and "target_roll_mean_96" in df.columns:
        df["roll_ratio_16_96"] = df["target_roll_mean_16"] / (df["target_roll_mean_96"] + 1e-6)
    if "target_roll_mean_48" in df.columns and "target_roll_mean_336" in df.columns:
        df["roll_ratio_48_336"] = df["target_roll_mean_48"] / (df["target_roll_mean_336"] + 1e-6)
    if "target_ema_8" in df.columns and "target_ema_96" in df.columns:
        df["ema_ratio_8_96"] = df["target_ema_8"] / (df["target_ema_96"] + 1e-6)
    return df


def train_lgb_models(X_fit, y_fit, X_valid, y_valid, params):
    models = {}
    for step_col in FUTURE_TARGET_COLS:
        m = lgb.LGBMRegressor(**params)
        m.fit(
            X_fit, y_fit[step_col],
            eval_set=[(X_valid, y_valid[step_col])],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        models[step_col] = m
        best = next(iter(m.best_score_.get("valid_0", {}).values()), float("nan"))
        print(f"    {step_col:20s}  iter={m.best_iteration_:4d}  score={best:.4f}")
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

for label, seeds, cs, ss, ff_bynode in GROUPS:
    print(f"\n>>> {label}: cs={cs} ss={ss} bynode={ff_bynode} <<<")
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
            "colsample_bytree": cs,
            "subsample": ss,
            "feature_fraction_bynode": ff_bynode,
        }
        models = train_lgb_models(X_fit, y_fit, X_valid, y_valid, params)
        vp = predict_steps(models, X_valid)
        tp = predict_steps(models, X_test)
        w, r, t = metric.calculate_components(y_valid, vp)
        print(f"  {label} seed={seed}: WAPE={w:.4f} RBias={r:.4f} Total={t:.4f}  [{(datetime.now()-t0).seconds}s]")
        sys.stdout.flush()
        vp_list.append(vp)
        tp_list.append(tp)
        joblib.dump(models, models_dir / f"lgb_{label}_seed{seed}.pkl")


n_total = len(vp_list)
print(f"\n[Ensemble] Averaging {n_total} predictions...")
sys.stdout.flush()
valid_ens = pd.DataFrame(
    np.mean([p.values for p in vp_list], axis=0),
    columns=FUTURE_TARGET_COLS,
    index=X_valid.index,
).clip(lower=0)
test_ens = pd.DataFrame(
    np.mean([p.values for p in tp_list], axis=0),
    columns=FUTURE_TARGET_COLS,
    index=X_test.index,
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

# Save submission
sub = build_submission(test_cal, X_test, inference_ts, test_df)
out_sub = f"submission_team_{EXPERIMENT_NAME}.csv"
sub.to_csv(out_sub, index=False)
print(f"Saved: {out_sub}  ({len(sub)} rows)")

# Update experiments.json
exp_path = Path("experiments.json")
exps = json.loads(exp_path.read_text(encoding="utf-8"))
exps.append({
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "name": EXPERIMENT_NAME,
    "params": {
        "groups": [(g[0], g[1], g[2], g[3], g[4]) for g in GROUPS],
        "alpha": ALPHA,
        "num_leaves": 511,
        "n_models": n_total,
        "train_days": 21,
        "features": "158 (154 + 4 roll-ratio)",
        "calibration": "per_step",
    },
    "wape": round(w_cal, 6),
    "rbias": round(r_cal, 6),
    "total": round(t_cal, 6),
    "note": (
        f"GroupA(cs=0.5,bynode=0.7) + GroupB(cs=1.0,bynode=0.5). "
        f"Raw: WAPE={w_raw:.4f} RBias={r_raw:.4f} Total={t_raw:.4f}. "
        f"Cal: total={t_cal:.4f}. "
        f"Cal factors: {list(factors.values())}"
    ),
})
exp_path.write_text(json.dumps(exps, indent=2, ensure_ascii=False), encoding="utf-8")

print("\n" + "=" * 60)
print(f"EXPERIMENT {EXPERIMENT_NAME} COMPLETE")
print(f"  feature_fraction_bynode diversity test")
print(f"  Raw:        WAPE={w_raw:.4f}  RBias={r_raw:.4f}  Total={t_raw:.4f}")
print(f"  Calibrated: WAPE={w_cal:.4f}  RBias={r_cal:.4f}  Total={t_cal:.4f}")
print(f"  vs exp57 baseline: 0.239609, exp53: 0.239683")
print("=" * 60)
