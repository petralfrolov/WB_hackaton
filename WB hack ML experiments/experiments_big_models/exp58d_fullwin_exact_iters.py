"""Experiment 58d: Full-window, exact exp51 iters (no ×1.25), cal51 factors.

Key differences from exp58:
  - n_estimators = exact exp51 mean best_iteration per step (no multiplier)
  - calibration = exp51 factors (not biased valid)
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

EXPERIMENT_NAME = "exp58d_fullwin_exact_iters"
ALPHA = 0.55

# Mean best_iteration across 10 seeds in exp51 (no multiplier)
STEP_ITERS = {
    "target_step_1":  1041,
    "target_step_2":  648,
    "target_step_3":  377,
    "target_step_4":  284,
    "target_step_5":  215,
    "target_step_6":  191,
    "target_step_7":  189,
    "target_step_8":  175,
    "target_step_9":  173,
    "target_step_10": 185,
}

# exp51 calibration factors
CAL51 = {
    "target_step_1":  1.0176,
    "target_step_2":  1.0184,
    "target_step_3":  1.0185,
    "target_step_4":  1.0185,
    "target_step_5":  1.0199,
    "target_step_6":  1.0214,
    "target_step_7":  1.0212,
    "target_step_8":  1.0217,
    "target_step_9":  1.0218,
    "target_step_10": 1.0218,
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
print("  Full 21d window | exact exp51 iters | cal51 factors")
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

X_fit, y_fit, X_valid, y_valid, X_test, inference_ts, _ = split_data(train_df, feature_cols, train_days=21)
X_fit, X_valid, X_test, _ = encode_categoricals(X_fit, X_valid, X_test, feature_cols)

# Train on full 21d window (fit + valid)
X_train = pd.concat([X_fit, X_valid], ignore_index=True)
y_train = pd.concat([y_fit, y_valid], ignore_index=True)
print(f"Full train={len(X_train):,}  (fit={len(X_fit):,} + valid={len(X_valid):,})")
sys.stdout.flush()

models_dir = Path("models") / EXPERIMENT_NAME
models_dir.mkdir(parents=True, exist_ok=True)
tp_list = []

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
            print(f"    {step_col:20s}  iters={n_iters:4d}")
            sys.stdout.flush()
        tp = predict_steps(models, X_test)
        tp_list.append(tp)
        print(f"  {label} seed={seed} done  [{(datetime.now()-t0).seconds}s]")
        sys.stdout.flush()
        joblib.dump(models, models_dir / f"lgb_{label}_seed{seed}.pkl")

n_total = len(tp_list)
print(f"\n[Ensemble] Averaging {n_total} predictions...")
test_ens = pd.DataFrame(
    np.mean([p.values for p in tp_list], axis=0),
    columns=FUTURE_TARGET_COLS, index=X_test.index
).clip(lower=0)

# Apply exp51 calibration factors
test_cal = test_ens.copy()
for col in FUTURE_TARGET_COLS:
    test_cal[col] = (test_ens[col] * CAL51[col]).clip(lower=0)

print(f"[Cal51 factors]: {list(CAL51.values())}")
sys.stdout.flush()

sub = build_submission(test_cal, X_test, inference_ts, test_df)
sub.to_csv(f"submission_team_{EXPERIMENT_NAME}.csv", index=False)
print(f"Saved: submission_team_{EXPERIMENT_NAME}.csv  ({len(sub)} rows)")

exp_path = Path("experiments.json")
exps = json.loads(exp_path.read_text(encoding="utf-8"))
exps.append({"timestamp": datetime.now().isoformat(timespec="seconds"), "name": EXPERIMENT_NAME,
    "params": {"groups": [(g[0], g[1], g[2], g[3]) for g in GROUPS], "alpha": ALPHA,
               "num_leaves": 511, "n_models": n_total, "train_days": 21,
               "calibration": "cal51_external", "mode": "full_window_exact_iters",
               "step_iters": STEP_ITERS},
    "note": "Full 21d window, exact exp51 mean iters, cal51 factors. No local metric."})
exp_path.write_text(json.dumps(exps, indent=2, ensure_ascii=False), encoding="utf-8")

print("\n" + "=" * 60)
print(f"EXPERIMENT {EXPERIMENT_NAME} COMPLETE")
print("  *** No local metric — orient on PUBLIC LB ***")
print("=" * 60)
