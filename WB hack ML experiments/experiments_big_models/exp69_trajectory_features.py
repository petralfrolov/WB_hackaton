"""Experiment 69: More trajectory features — lag acceleration + slope features.

exp61 roll-ratio features helped on public LB (+0.0001). Let's push further
with more trajectory/gradient features that capture demand dynamics:

New features (6 total):
  1. lag_accel_48    = target_lag_1 - 2*target_lag_48  + target_lag_96
     → 2nd discrete derivative at 1/48/96 lags (demand acceleration)
  2. lag_accel_336   = target_lag_48 - 2*target_lag_336 + target_lag_672
     → weekly acceleration (is weekly pattern strengthening/weakening?)
  3. roll_slope_48   = (target_roll_mean_8 - target_roll_mean_48) / 40
     → linear slope of rolling means (short-to-medium demand trend direction)
  4. roll_slope_96   = (target_roll_mean_16 - target_roll_mean_96) / 80
     → medium-to-long slope
  5. ema_slope       = target_ema_8 - target_ema_96
     → absolute EMA divergence (complement to ema_ratio_8_96)
  6. recent_vs_seasonal = target_roll_mean_48 - route_slot_mean
     → is today's rolling avg above/below this route's typical level for this slot?

Config: same GroupB (cs=0.5, ss=0.7, leaves=511, 10 seeds, alpha=0.55, 21d).
Total features: ~164 (158 + 6 new).
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

EXPERIMENT_NAME = "exp69_trajectory_features"
SEEDS = [42, 123, 456, 789, 1234, 2024, 7, 314, 99, 888]
ALPHA = 0.55
COLSAMPLE = 0.5
SUBSAMPLE = 0.7

BASE_LGB = dict(n_estimators=5000, learning_rate=0.05, num_leaves=511,
    min_child_samples=10, subsample_freq=1, reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1, verbose=-1)

print("=" * 60)
print(f"EXPERIMENT: {EXPERIMENT_NAME}")
print(f"  10 seeds | cs={COLSAMPLE} ss={SUBSAMPLE} | leaves=511 | alpha={ALPHA} | 21d")
print("  NEW: lag_accel_48/336, roll_slope_48/96, ema_slope, recent_vs_seasonal")
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
    """NEW: 2nd-order lag derivatives, slopes, recent-vs-seasonal deviation."""
    # Lag acceleration: 2nd discrete derivative
    if all(c in df.columns for c in ["target_lag_1", "target_lag_48", "target_lag_96"]):
        df["lag_accel_48"] = df["target_lag_1"] - 2 * df["target_lag_48"] + df["target_lag_96"]

    if all(c in df.columns for c in ["target_lag_48", "target_lag_336", "target_lag_672"]):
        df["lag_accel_336"] = df["target_lag_48"] - 2 * df["target_lag_336"] + df["target_lag_672"]

    # Rolling slope: (short_mean - long_mean) / window_gap
    if "target_roll_mean_8" in df.columns and "target_roll_mean_48" in df.columns:
        df["roll_slope_48"] = (df["target_roll_mean_8"] - df["target_roll_mean_48"]) / 40.0

    if "target_roll_mean_16" in df.columns and "target_roll_mean_96" in df.columns:
        df["roll_slope_96"] = (df["target_roll_mean_16"] - df["target_roll_mean_96"]) / 80.0

    # EMA absolute divergence
    if "target_ema_8" in df.columns and "target_ema_96" in df.columns:
        df["ema_slope"] = df["target_ema_8"] - df["target_ema_96"]

    # Recent rolling avg vs seasonal mean for this route/slot
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

for seed in SEEDS:
    t0 = datetime.now()
    print(f"\n--- cs={COLSAMPLE} ss={SUBSAMPLE} seed={seed} [{t0.strftime('%H:%M:%S')}] ---")
    sys.stdout.flush()
    params = {**BASE_LGB, "objective": "quantile", "alpha": ALPHA,
              "random_state": seed, "colsample_bytree": COLSAMPLE, "subsample": SUBSAMPLE}
    models = train_lgb_models(X_fit, y_fit, X_valid, y_valid, params)
    vp = predict_steps(models, X_valid)
    tp = predict_steps(models, X_test)
    w, r, t = metric.calculate_components(y_valid, vp)
    print(f"  seed={seed}: WAPE={w:.4f} RBias={r:.4f} Total={t:.4f}  [{(datetime.now()-t0).seconds}s]")
    sys.stdout.flush()
    vp_list.append(vp); tp_list.append(tp)
    joblib.dump(models, models_dir / f"lgb_seed{seed}.pkl")

print(f"\n[Ensemble] Averaging {len(SEEDS)} predictions...")
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

new_feats = ["lag_accel_48", "lag_accel_336", "roll_slope_48", "roll_slope_96",
             "ema_slope", "recent_vs_seasonal"]
exp_path = Path("experiments.json")
exps = json.loads(exp_path.read_text(encoding="utf-8"))
exps.append({"timestamp": datetime.now().isoformat(timespec="seconds"), "name": EXPERIMENT_NAME,
    "params": {"seeds": SEEDS, "alpha": ALPHA, "num_leaves": 511,
               "colsample_bytree": COLSAMPLE, "subsample": SUBSAMPLE,
               "n_models": len(SEEDS), "train_days": 21,
               "features": f"158+6=164 (added {new_feats})", "calibration": "per_step"},
    "wape": round(w_cal,6), "rbias": round(r_cal,6), "total": round(t_cal,6),
    "note": (f"6 new trajectory feats: {new_feats}. "
             f"cs=0.5 ss=0.7 leaves=511 10 seeds. "
             f"Raw:{t_raw:.4f} Cal:{t_cal:.4f}. Factors:{list(factors.values())}")})
exp_path.write_text(json.dumps(exps, indent=2, ensure_ascii=False), encoding="utf-8")

print("\n" + "=" * 60)
print(f"EXPERIMENT {EXPERIMENT_NAME} COMPLETE")
print(f"  New features: {new_feats}")
print(f"  Raw:        WAPE={w_raw:.4f}  RBias={r_raw:.4f}  Total={t_raw:.4f}")
print(f"  Calibrated: WAPE={w_cal:.4f}  RBias={r_cal:.4f}  Total={t_cal:.4f}")
print(f"  vs exp61: 0.2400 local / 0.2501 public")
print("=" * 60)
