"""Experiment 70: roll-ratio + structural diversity + trajectory features (best combo).

Combines everything that showed improvement:
  - exp61/66/67: roll-ratio features improve public LB generalization
  - exp69: trajectory features (acceleration, slope, recent_vs_seasonal)
  - exp51/66: structural cs-diversity helps on public LB

This is a convergence experiment: use the richest feature set (164 feats)
with the structural diversity that gave best public LB (exp51 config).

Config = exp51 structural diversity + all 10 winning extra features:
  Group A (4 seeds): cs=0.8, ss=0.8
  Group B (3 seeds): cs=0.5, ss=0.7
  Group C (3 seeds): cs=0.7, ss=0.9
  All: leaves=511, alpha=0.55, 21d, 164 features
"""

import json, sys
from datetime import datetime
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from config import FUTURE_TARGET_COLS
from data import build_feature_cols, create_future_targets, encode_categoricals, load_data, split_data
from features import make_features
from metrics import WapePlusRbias
from train import build_submission, predict_steps

EXPERIMENT_NAME = "exp70_full_features_struct_diversity"
ALPHA = 0.55
GROUPS = [
    ("A_cs08", [42, 123, 456, 789],  0.8, 0.8),
    ("B_cs05", [1234, 2024, 7],      0.5, 0.7),
    ("C_cs07", [314, 99, 888],       0.7, 0.9),
]
BASE_LGB = dict(n_estimators=5000, learning_rate=0.05, num_leaves=511,
    min_child_samples=10, subsample_freq=1, reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1, verbose=-1)

print("=" * 60)
print(f"EXPERIMENT: {EXPERIMENT_NAME}")
print("  exp51 config + ALL extra features (roll-ratio + trajectory = 164 feats)")
print("  GroupA(cs=0.8) + GroupB(cs=0.5) + GroupC(cs=0.7), 21d, alpha=0.55")
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


def add_trajectory_features(df):
    if all(c in df.columns for c in ["target_lag_1", "target_lag_48", "target_lag_96"]):
        df["lag_accel_48"] = df["target_lag_1"] - 2 * df["target_lag_48"] + df["target_lag_96"]
    if all(c in df.columns for c in ["target_lag_48", "target_lag_336", "target_lag_672"]):
        df["lag_accel_336"] = df["target_lag_48"] - 2 * df["target_lag_336"] + df["target_lag_672"]
    if "target_roll_mean_8" in df.columns and "target_roll_mean_48" in df.columns:
        df["roll_slope_48"] = (df["target_roll_mean_8"] - df["target_roll_mean_48"]) / 40.0
    if "target_roll_mean_16" in df.columns and "target_roll_mean_96" in df.columns:
        df["roll_slope_96"] = (df["target_roll_mean_16"] - df["target_roll_mean_96"]) / 80.0
    if "target_ema_8" in df.columns and "target_ema_96" in df.columns:
        df["ema_slope"] = df["target_ema_8"] - df["target_ema_96"]
    if "target_roll_mean_48" in df.columns and "route_slot_mean" in df.columns:
        df["recent_vs_seasonal"] = df["target_roll_mean_48"] - df["route_slot_mean"]
    return df


def train_lgb_models(X_fit, y_fit, X_valid, y_valid, params):
    models = {}
    for step_col in FUTURE_TARGET_COLS:
        m = lgb.LGBMRegressor(**params)
        m.fit(X_fit, y_fit[step_col], eval_set=[(X_valid, y_valid[step_col])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
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
train_df = add_trajectory_features(train_df)
train_df = create_future_targets(train_df)
feature_cols = build_feature_cols(train_df)
print(f"Feature count: {len(feature_cols)}")
sys.stdout.flush()

X_fit, y_fit, X_valid, y_valid, X_test, inference_ts, _ = split_data(train_df, feature_cols, train_days=21)
X_fit, X_valid, X_test, _ = encode_categoricals(X_fit, X_valid, X_test, feature_cols)
print(f"Fit={len(X_fit):,}  Valid={len(X_valid):,}")
sys.stdout.flush()

metric = WapePlusRbias()
models_dir = Path("models") / EXPERIMENT_NAME
models_dir.mkdir(parents=True, exist_ok=True)
vp_list, tp_list = [], []

for label, seeds, cs, ss in GROUPS:
    print(f"\n>>> {label}: cs={cs} ss={ss} <<<")
    sys.stdout.flush()
    for seed in seeds:
        t0 = datetime.now()
        print(f"\n--- {label} seed={seed} [{t0.strftime('%H:%M:%S')}] ---")
        sys.stdout.flush()
        params = {**BASE_LGB, "objective": "quantile", "alpha": ALPHA,
                  "random_state": seed, "colsample_bytree": cs, "subsample": ss}
        models = train_lgb_models(X_fit, y_fit, X_valid, y_valid, params)
        vp = predict_steps(models, X_valid)
        tp = predict_steps(models, X_test)
        w, r, t = metric.calculate_components(y_valid, vp)
        print(f"  {label} seed={seed}: WAPE={w:.4f} RBias={r:.4f} Total={t:.4f}  [{(datetime.now()-t0).seconds}s]")
        sys.stdout.flush()
        vp_list.append(vp); tp_list.append(tp)
        joblib.dump(models, models_dir / f"lgb_{label}_seed{seed}.pkl")

n_total = len(vp_list)
print(f"\n[Ensemble] Averaging {n_total} predictions...")
sys.stdout.flush()
valid_ens = pd.DataFrame(np.mean([p.values for p in vp_list], axis=0),
    columns=FUTURE_TARGET_COLS, index=X_valid.index).clip(lower=0)
test_ens = pd.DataFrame(np.mean([p.values for p in tp_list], axis=0),
    columns=FUTURE_TARGET_COLS, index=X_test.index).clip(lower=0)
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
exps.append({"timestamp": datetime.now().isoformat(timespec="seconds"), "name": EXPERIMENT_NAME,
    "params": {"groups": [(g[0],g[1],g[2],g[3]) for g in GROUPS], "alpha": ALPHA,
               "num_leaves": 511, "n_models": n_total, "train_days": 21,
               "features": "164 (154+4 roll-ratio+6 trajectory)", "calibration": "per_step"},
    "wape": round(w_cal,6), "rbias": round(r_cal,6), "total": round(t_cal,6),
    "note": (f"exp51-config + ALL extra feats (roll-ratio+trajectory=164). "
             f"GroupA(cs=0.8)+GroupB(cs=0.5)+GroupC(cs=0.7). "
             f"Raw:{t_raw:.4f} Cal:{t_cal:.4f}. Factors:{list(factors.values())}")})
exp_path.write_text(json.dumps(exps, indent=2, ensure_ascii=False), encoding="utf-8")

print("\n" + "=" * 60)
print(f"EXPERIMENT {EXPERIMENT_NAME} COMPLETE")
print(f"  Full feature set (164): roll-ratio + trajectory + structural diversity")
print(f"  Raw:        WAPE={w_raw:.4f}  RBias={r_raw:.4f}  Total={t_raw:.4f}")
print(f"  Calibrated: WAPE={w_cal:.4f}  RBias={r_cal:.4f}  Total={t_cal:.4f}")
print(f"  vs exp66 (same config, 158 feats), vs exp69 (164 feats, pure cs=0.5)")
print("=" * 60)
