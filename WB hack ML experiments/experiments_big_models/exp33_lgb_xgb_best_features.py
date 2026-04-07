"""Experiment 33: LGB + XGB hybrid with 154 best features.

Hypothesis: mixing LGB and XGB diversity (like exp28) with the best 154 features
(like exp_best_features_full) should outperform either alone:
  - exp28 (3LGB+3XGB, base features): val=0.2402
  - exp_best (10LGB, 154 feats):        val=0.2405
  => combined diversity + best features -> target < 0.2400

Architecture:
  - 3 LGB seeds (42,123,456) alpha=0.52
  - 3 LGB seeds (42,123,456) alpha=0.55
  - 2 XGB seeds (42,123)     alpha=0.52
  - 2 XGB seeds (42,123)     alpha=0.55
  = 10 models total, per-step calibration
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBRegressor
import joblib
import json
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, ".")

from config import TRACK, FUTURE_TARGET_COLS, TARGET_COL
from data import load_data, create_future_targets, build_feature_cols, split_data, encode_categoricals
from features import make_features
from metrics import WapePlusRbias
from train import train_lgb_models, predict_steps, build_submission

EXPERIMENT_NAME = "exp33_lgb_xgb_best_features"
LGB_SEEDS = [42, 123, 456]
XGB_SEEDS = [42, 123]
ALPHAS = [0.52, 0.55]

BASE_LGB = dict(
    n_estimators=5000, learning_rate=0.05, num_leaves=127,
    min_child_samples=20, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    n_jobs=-1, verbose=-1,
)

BASE_XGB = dict(
    n_estimators=5000, learning_rate=0.05, max_depth=7,
    min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    objective="reg:quantileerror",
    tree_method="hist", device="cpu",
    n_jobs=-1, early_stopping_rounds=100,
)

print("=" * 60)
print(f"EXPERIMENT: {EXPERIMENT_NAME}")
print(f"  100% routes | 3LGB*2alpha + 2XGB*2alpha = 10 models | per-step calib")
print(f"  154 features (extended + rolling_cv + lag_ratios)")
print("=" * 60)
sys.stdout.flush()


def add_winning_features(df):
    """Add the 7 winning features: rolling_cv + lag_ratios."""
    for w in [4, 8, 48]:
        std_col = f"target_roll_std_{w}"
        mean_col = f"target_roll_mean_{w}"
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


# ── Load & feature engineering ────────────────────────────────────────────────
print("Loading data...")
sys.stdout.flush()
train_df, test_df = load_data()
print(f"Routes: {train_df['route_id'].nunique()} (full) | Rows: {len(train_df)}")
sys.stdout.flush()

print("Building features (extended + winning)...")
sys.stdout.flush()
train_df = make_features(train_df, extended=True)
train_df = add_winning_features(train_df)
train_df = create_future_targets(train_df)
feature_cols = build_feature_cols(train_df)
print(f"Feature count: {len(feature_cols)}")
sys.stdout.flush()

X_fit, y_fit, X_valid, y_valid, X_test, inference_ts, test_model_df = split_data(train_df, feature_cols)
X_fit, X_valid, X_test, cat_features = encode_categoricals(X_fit, X_valid, X_test, feature_cols)

# XGB needs integer-coded categoricals
X_fit_xgb = X_fit.copy()
X_valid_xgb = X_valid.copy()
X_test_xgb = X_test.copy()
for col in cat_features:
    X_fit_xgb[col] = X_fit_xgb[col].cat.codes.astype(np.int32)
    X_valid_xgb[col] = X_valid_xgb[col].cat.codes.astype(np.int32)
    X_test_xgb[col] = X_test_xgb[col].cat.codes.astype(np.int32)

metric = WapePlusRbias()
models_dir = Path("models") / EXPERIMENT_NAME
models_dir.mkdir(parents=True, exist_ok=True)
print(f"Models -> {models_dir}")
sys.stdout.flush()

valid_preds_list = []
test_preds_list = []

# ── LGB models ────────────────────────────────────────────────────────────────
for alpha in ALPHAS:
    print(f"\n--- LGB alpha={alpha} ---")
    sys.stdout.flush()
    for seed in LGB_SEEDS:
        t0 = datetime.now()
        print(f"  seed={seed} [{t0.strftime('%H:%M:%S')}]")
        sys.stdout.flush()
        p = {**BASE_LGB, "objective": "quantile", "alpha": alpha, "random_state": seed}
        models = train_lgb_models(X_fit, y_fit, X_valid, y_valid, lgb_params=p)
        vp = predict_steps(models, X_valid)
        tp = predict_steps(models, X_test)
        w, r, t = metric.calculate_components(y_valid, vp)
        elapsed = (datetime.now() - t0).seconds
        print(f"  seed={seed} alpha={alpha}: WAPE={w:.4f} RBias={r:.4f} Total={t:.4f}  [{elapsed}s]")
        sys.stdout.flush()
        valid_preds_list.append(vp)
        test_preds_list.append(tp)
        path = models_dir / f"lgb_a{int(alpha*100):03d}_seed{seed}.pkl"
        joblib.dump(models, path)
        print(f"  Saved: {path}")
        sys.stdout.flush()

# ── XGB models ────────────────────────────────────────────────────────────────
for alpha in ALPHAS:
    print(f"\n--- XGB alpha={alpha} ---")
    sys.stdout.flush()
    for seed in XGB_SEEDS:
        t0 = datetime.now()
        print(f"  XGB seed={seed} [{t0.strftime('%H:%M:%S')}]")
        sys.stdout.flush()
        p = {**BASE_XGB, "quantile_alpha": alpha, "random_state": seed}
        models_xgb = {}
        for step_col in FUTURE_TARGET_COLS:
            m = XGBRegressor(**p)
            m.fit(
                X_fit_xgb, y_fit[step_col],
                eval_set=[(X_valid_xgb, y_valid[step_col])],
                verbose=False
            )
            models_xgb[step_col] = m
            print(f"    {step_col:20s}  best_iter={m.best_iteration:4d}")
            sys.stdout.flush()
        vp = pd.DataFrame(
            {sc: np.clip(models_xgb[sc].predict(X_valid_xgb), 0, None) for sc in FUTURE_TARGET_COLS},
            index=X_valid.index
        )
        tp = pd.DataFrame(
            {sc: np.clip(models_xgb[sc].predict(X_test_xgb), 0, None) for sc in FUTURE_TARGET_COLS},
            index=X_test.index
        )
        w, r, t = metric.calculate_components(y_valid, vp)
        elapsed = (datetime.now() - t0).seconds
        print(f"  XGB seed={seed} alpha={alpha}: WAPE={w:.4f} RBias={r:.4f} Total={t:.4f}  [{elapsed}s]")
        sys.stdout.flush()
        valid_preds_list.append(vp)
        test_preds_list.append(tp)
        xgb_dir = models_dir / f"xgb_a{int(alpha*100):03d}_seed{seed}"
        xgb_dir.mkdir(exist_ok=True)
        for sc, m in models_xgb.items():
            m.save_model(str(xgb_dir / f"{sc}.ubj"))
        print(f"  Saved: {xgb_dir}/")
        sys.stdout.flush()

# ── Ensemble ──────────────────────────────────────────────────────────────────
n_models = len(valid_preds_list)
print(f"\n[Ensemble] Averaging {n_models} predictions...")
sys.stdout.flush()

valid_ens = pd.DataFrame(
    np.mean([p.values for p in valid_preds_list], axis=0),
    columns=FUTURE_TARGET_COLS, index=X_valid.index
).clip(lower=0)

test_ens = pd.DataFrame(
    np.mean([p.values for p in test_preds_list], axis=0),
    columns=FUTURE_TARGET_COLS, index=X_test.index
).clip(lower=0)

w_raw, r_raw, t_raw = metric.calculate_components(y_valid, valid_ens)
print(f"[Raw ensemble]  WAPE={w_raw:.4f} RBias={r_raw:.4f} Total={t_raw:.4f}")
sys.stdout.flush()

# ── Per-step calibration ──────────────────────────────────────────────────────
print("[Per-step calibration]")
sys.stdout.flush()
valid_cal = valid_ens.copy()
test_cal = test_ens.copy()
for c in FUTURE_TARGET_COLS:
    f = float(y_valid[c].sum()) / max(float(valid_ens[c].sum()), 1e-9)
    valid_cal[c] = (valid_ens[c] * f).clip(lower=0)
    test_cal[c] = (test_ens[c] * f).clip(lower=0)

w_cal, r_cal, t_cal = metric.calculate_components(y_valid, valid_cal)
print(f"[Calibrated]    WAPE={w_cal:.4f} RBias={r_cal:.4f} Total={t_cal:.4f}")
sys.stdout.flush()

# ── Submission ────────────────────────────────────────────────────────────────
print("\n[Submission] Building...")
sys.stdout.flush()
sub = build_submission(test_cal, X_test, inference_ts, test_df)
sub_path = f"submission_{TRACK}_{EXPERIMENT_NAME}.csv"
sub.to_csv(sub_path, index=False)
print(f"Saved: {sub_path}  ({len(sub)} rows)")
sys.stdout.flush()

# ── Log ───────────────────────────────────────────────────────────────────────
exp_path = Path("experiments.json")
try:
    experiments = json.loads(exp_path.read_text(encoding="utf-8"))
except Exception:
    experiments = []

record = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "name": EXPERIMENT_NAME,
    "params": {
        "lgb_seeds": LGB_SEEDS,
        "xgb_seeds": XGB_SEEDS,
        "alphas": ALPHAS,
        "n_models": n_models,
        "features": "extended+rolling_cv+lag_ratios (154)",
        "calibration": "per_step",
        "models": f"{len(LGB_SEEDS)*len(ALPHAS)}LGB+{len(XGB_SEEDS)*len(ALPHAS)}XGB",
    },
    "wape": round(w_cal, 6),
    "rbias": round(r_cal, 6),
    "total": round(t_cal, 6),
    "note": (
        f"LGB+XGB hybrid, 154 feats. "
        f"Raw: WAPE={w_raw:.4f} RBias={r_raw:.4f} Total={t_raw:.4f}; "
        f"Calibrated: WAPE={w_cal:.4f} RBias={r_cal:.4f} Total={t_cal:.4f}"
    ),
}
experiments.append(record)
exp_path.write_text(json.dumps(experiments, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Logged -> {exp_path}")

print("\n" + "=" * 60)
print(f"EXPERIMENT {EXPERIMENT_NAME} COMPLETE")
print(f"  Calibrated: WAPE={w_cal:.4f}  RBias={r_cal:.4f}  Total={t_cal:.4f}")
print("=" * 60)
