"""Experiment 58: Full-window training (no held-out validation).

Standard approach: 80% fit + 20% valid, valid used only for early stopping.
This throws away the freshest 4 days before test.

This exp: train on ALL 21 days (fit+valid) with fixed n_estimators per step
derived from exp51 best_iterations × 1.25 (more data → more iters).

Groups: exp51 proven structure — A(cs=0.8,ss=0.8) + B(cs=0.5,ss=0.7) + C(cs=0.7,ss=0.9)

Note: no unbiased local metric — orient on public LB.
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

EXPERIMENT_NAME = "exp58_full_window"
ALPHA = 0.55

# Fixed iterations per step: exp51 best_iteration mean across seeds × 1.25 (more data)
STEP_ITERS = {
    "target_step_1":  1100,
    "target_step_2":  750,
    "target_step_3":  450,
    "target_step_4":  350,
    "target_step_5":  300,
    "target_step_6":  300,
    "target_step_7":  300,
    "target_step_8":  300,
    "target_step_9":  300,
    "target_step_10": 300,
}

# exp51 proven group structure
GROUPS = [
    ("A_cs08", [42, 123, 456, 789],  0.8, 0.8),
    ("B_cs05", [1234, 2024, 7],      0.5, 0.7),
    ("C_cs07", [314, 99, 888],       0.7, 0.9),
]

BASE_LGB = dict(learning_rate=0.05, num_leaves=511, min_child_samples=10,
    subsample_freq=1, reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1, verbose=-1)

print("=" * 60)
print(f"EXPERIMENT: {EXPERIMENT_NAME}")
print("  Full 21d window (no early stopping) | exp51 groups | per-step iters")
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

print("Loading + building features...")
sys.stdout.flush()
train_df_raw, test_df = load_data()
train_df = make_features(train_df_raw, extended=True)
train_df = add_winning_features(train_df)
train_df = create_future_targets(train_df)
feature_cols = build_feature_cols(train_df)
print(f"Feature count: {len(feature_cols)}")
sys.stdout.flush()

# Split normally to get valid/test structure — but we'll merge fit+valid for training
X_fit, y_fit, X_valid, y_valid, X_test, inference_ts, _ = split_data(train_df, feature_cols, train_days=21)
X_fit, X_valid, X_test, _ = encode_categoricals(X_fit, X_valid, X_test, feature_cols)

# Merge fit+valid → full training set
X_train = pd.concat([X_fit, X_valid], ignore_index=True)
y_train = pd.concat([y_fit, y_valid], ignore_index=True)
print(f"Full train={len(X_train):,}  (fit={len(X_fit):,} + valid={len(X_valid):,})")
sys.stdout.flush()

metric = WapePlusRbias()
models_dir = Path("models") / EXPERIMENT_NAME
models_dir.mkdir(parents=True, exist_ok=True)
tp_list, vp_list = [], []

for label, seeds, cs, ss in GROUPS:
    print(f"\n>>> {label}: {len(seeds)} seeds | cs={cs} ss={ss} <<<")
    sys.stdout.flush()
    for seed in seeds:
        t0 = datetime.now()
        print(f"\n--- {label} seed={seed} [{t0.strftime('%H:%M:%S')}] ---")
        sys.stdout.flush()
        models = {}
        for step_col in FUTURE_TARGET_COLS:
            n_iters = STEP_ITERS[step_col]
            params = {**BASE_LGB, "objective": "quantile", "alpha": ALPHA,
                      "random_state": seed, "colsample_bytree": cs, "subsample": ss,
                      "n_estimators": n_iters}
            m = lgb.LGBMRegressor(**params)
            m.fit(X_train, y_train[step_col])
            models[step_col] = m
            print(f"    {step_col:20s}  iters={n_iters:4d}  (fixed)")
            sys.stdout.flush()
        vp = predict_steps(models, X_valid)
        tp = predict_steps(models, X_test)
        # Valid score is biased (trained on valid) — sanity check only
        w, r, t = metric.calculate_components(y_valid, vp)
        print(f"  {label} seed={seed}: WAPE={w:.4f} RBias={r:.4f} Total={t:.4f} [biased!]  [{(datetime.now()-t0).seconds}s]")
        sys.stdout.flush()
        vp_list.append(vp); tp_list.append(tp)
        joblib.dump(models, models_dir / f"lgb_{label}_seed{seed}.pkl")

n_total = len(tp_list)
print(f"\n[Ensemble] Averaging {n_total} predictions...")
sys.stdout.flush()
valid_ens = pd.DataFrame(np.mean([p.values for p in vp_list], axis=0), columns=FUTURE_TARGET_COLS, index=X_valid.index).clip(lower=0)
test_ens  = pd.DataFrame(np.mean([p.values for p in tp_list], axis=0), columns=FUTURE_TARGET_COLS, index=X_test.index).clip(lower=0)
w_raw, r_raw, t_raw = metric.calculate_components(y_valid, valid_ens)
print(f"[Raw biased valid] WAPE={w_raw:.4f} RBias={r_raw:.4f} Total={t_raw:.4f} (not comparable to other exps)")

# Calibration: use valid factors even though biased — best we can do without held-out set
factors = {}
valid_cal, test_cal = valid_ens.copy(), test_ens.copy()
for col in FUTURE_TARGET_COLS:
    f = float(y_valid[col].sum()) / max(float(valid_ens[col].sum()), 1e-9)
    valid_cal[col] = (valid_ens[col] * f).clip(lower=0)
    test_cal[col]  = (test_ens[col]  * f).clip(lower=0)
    factors[col] = round(f, 4)
w_cal, r_cal, t_cal = metric.calculate_components(y_valid, valid_cal)
print(f"[Cal biased valid]  WAPE={w_cal:.4f} RBias={r_cal:.4f} Total={t_cal:.4f}")
print(f"[Cal factors]: {list(factors.values())}")
sys.stdout.flush()

sub = build_submission(test_cal, X_test, inference_ts, test_df)
sub.to_csv(f"submission_team_{EXPERIMENT_NAME}.csv", index=False)
print(f"Saved: submission_team_{EXPERIMENT_NAME}.csv  ({len(sub)} rows)")

exp_path = Path("experiments.json")
exps = json.loads(exp_path.read_text(encoding="utf-8"))
exps.append({"timestamp": datetime.now().isoformat(timespec="seconds"), "name": EXPERIMENT_NAME,
    "params": {"groups": [(g[0], g[1], g[2], g[3]) for g in GROUPS], "alpha": ALPHA,
               "num_leaves": 511, "n_models": n_total, "train_days": 21,
               "calibration": "per_step_biased", "mode": "full_window_fixed_iters",
               "step_iters": STEP_ITERS},
    "wape": round(w_cal, 6), "rbias": round(r_cal, 6), "total": round(t_cal, 6),
    "note": "Full 21d window, no early stopping. Valid score BIASED. Check public LB."})
exp_path.write_text(json.dumps(exps, indent=2, ensure_ascii=False), encoding="utf-8")

print("\n" + "=" * 60)
print(f"EXPERIMENT {EXPERIMENT_NAME} COMPLETE")
print(f"  Biased valid: WAPE={w_cal:.4f}  RBias={r_cal:.4f}  Total={t_cal:.4f}")
print("  *** Orient on PUBLIC LB — local metric is not comparable ***")
print("=" * 60)
