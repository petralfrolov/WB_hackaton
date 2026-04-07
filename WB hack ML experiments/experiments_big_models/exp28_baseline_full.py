"""Experiment 28: Full-data baseline — best config on 100% routes.

Same best config as exp25/exp27 (LGB + XGB, alpha=0.52, per-step correction),
but on ALL routes (100% data) with 3 seeds each.
Establishes the full-data baseline score.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBRegressor
import json
import joblib
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, ".")

from config import TRACK, FUTURE_TARGET_COLS, TARGET_COL
from data import load_data, create_future_targets, build_feature_cols, split_data, encode_categoricals
from features import make_features
from metrics import WapePlusRbias
from train import train_lgb_models, predict_steps, compute_bias_factor, build_submission

EXPERIMENT_NAME = "exp28_baseline_full"
SEEDS = [42, 123, 456]

BASE_LGB = dict(
    n_estimators=5000, learning_rate=0.05, num_leaves=127,
    min_child_samples=20, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    objective="quantile", alpha=0.52, n_jobs=-1, verbose=-1,
)
BASE_XGB = dict(
    n_estimators=5000, learning_rate=0.05, max_depth=7,
    min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, objective="reg:quantileerror",
    quantile_alpha=0.52, tree_method="hist", device="cpu",
    n_jobs=-1, early_stopping_rounds=100,
)

print("=" * 60)
print(f"EXPERIMENT: {EXPERIMENT_NAME}")
print(f"  100% routes | {len(SEEDS)} seeds | LGB+XGB | alpha=0.52")
print("=" * 60)
sys.stdout.flush()

print("Loading data...")
sys.stdout.flush()
train_df, test_df = load_data()
all_routes = train_df["route_id"].unique()
print(f"Routes: {len(all_routes)} (full) | Train rows: {len(train_df)}")
sys.stdout.flush()

print("Building features...")
sys.stdout.flush()
train_df = make_features(train_df, extended=True)
train_df = create_future_targets(train_df)
feature_cols = build_feature_cols(train_df)
print(f"Feature count: {len(feature_cols)}")
sys.stdout.flush()

X_fit, y_fit, X_valid, y_valid, X_test, inference_ts, test_model_df = split_data(train_df, feature_cols)
X_fit, X_valid, X_test, cat_features = encode_categoricals(X_fit, X_valid, X_test, feature_cols)

X_fit_xgb = X_fit.copy()
X_valid_xgb = X_valid.copy()
for col in cat_features:
    X_fit_xgb[col] = X_fit_xgb[col].cat.codes.astype(np.int32)
    X_valid_xgb[col] = X_valid_xgb[col].cat.codes.astype(np.int32)

metric = WapePlusRbias()

def per_step_correct(pred_df, y_val):
    out = pred_df.copy()
    for c in FUTURE_TARGET_COLS:
        f = float(y_val[c].sum()) / max(float(pred_df[c].sum()), 1e-9)
        out[c] = (out[c] * f).clip(lower=0)
    return out

preds_lgb, preds_xgb = [], []
test_preds_lgb, test_preds_xgb = [], []

models_dir = Path("models") / EXPERIMENT_NAME
models_dir.mkdir(parents=True, exist_ok=True)
print(f"Models will be saved to: {models_dir}")
sys.stdout.flush()

print("\n--- LGB models ---")
sys.stdout.flush()
for seed in SEEDS:
    t0 = datetime.now()
    print(f"  LGB seed={seed} started at {t0.strftime('%H:%M:%S')} ...")
    sys.stdout.flush()
    p = {**BASE_LGB, "random_state": seed}
    models = train_lgb_models(X_fit, y_fit, X_valid, y_valid, lgb_params=p)
    vp = predict_steps(models, X_valid)
    w, r, t = metric.calculate_components(y_valid, vp)
    elapsed = (datetime.now() - t0).seconds
    print(f"  LGB seed={seed}: WAPE={w:.4f} RBias={r:.4f} Total={t:.4f}  [{elapsed}s]")
    sys.stdout.flush()
    preds_lgb.append(vp)
    test_preds_lgb.append(predict_steps(models, X_test))
    lgb_path = models_dir / f"lgb_seed{seed}.pkl"
    joblib.dump(models, lgb_path)
    print(f"  Saved: {lgb_path}")
    sys.stdout.flush()

print("\n--- XGB models ---")
sys.stdout.flush()
for seed in SEEDS:
    t0 = datetime.now()
    print(f"  XGB seed={seed} started at {t0.strftime('%H:%M:%S')} ...")
    sys.stdout.flush()
    p = {**BASE_XGB, "random_state": seed}
    models_xgb = {}
    X_test_xgb = X_test.copy()
    for col in cat_features:
        X_test_xgb[col] = X_test_xgb[col].cat.codes.astype(np.int32)
    for sc in FUTURE_TARGET_COLS:
        m = XGBRegressor(**p)
        m.fit(X_fit_xgb, y_fit[sc], eval_set=[(X_valid_xgb, y_valid[sc])], verbose=False)
        models_xgb[sc] = m
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
    print(f"  XGB seed={seed}: WAPE={w:.4f} RBias={r:.4f} Total={t:.4f}  [{elapsed}s]")
    sys.stdout.flush()
    preds_xgb.append(vp)
    test_preds_xgb.append(tp)
    xgb_dir = models_dir / f"xgb_seed{seed}"
    xgb_dir.mkdir(exist_ok=True)
    for sc, m in models_xgb.items():
        m.save_model(str(xgb_dir / f"{sc}.ubj"))
    print(f"  Saved: {xgb_dir}/")
    sys.stdout.flush()

ens = pd.DataFrame(
    np.mean([p.values for p in preds_lgb + preds_xgb], axis=0),
    columns=FUTURE_TARGET_COLS, index=X_valid.index
).clip(lower=0)

w_raw, r_raw, t_raw = metric.calculate_components(y_valid, ens)
corrected = per_step_correct(ens, y_valid)
w_c, r_c, t_c = metric.calculate_components(y_valid, corrected)
print(f"\n[Ensemble raw]       WAPE={w_raw:.4f} RBias={r_raw:.4f} Total={t_raw:.4f}")
print(f"[Ensemble per-step]  WAPE={w_c:.4f} RBias={r_c:.4f} Total={t_c:.4f}")
sys.stdout.flush()

# --- Log metrics ---
log_path = Path("experiments.json")
history = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
history.append({
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "name": EXPERIMENT_NAME,
    "params": dict(sample_frac=1.0, seeds=SEEDS, alpha=0.52,
                   models="3lgb+3xgb", calibration="per_step"),
    "wape": round(t_c, 6), "rbias": 0.0, "total": round(t_c, 6),
})
log_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n[BASELINE FULL] VALID total = {t_c:.6f}")
print("Logged.")
sys.stdout.flush()

# --- Submission ---
print("\n[Submission] Building test predictions...")
sys.stdout.flush()
bias_factor = compute_bias_factor(y_valid, ens)
test_ens = pd.DataFrame(
    np.mean([p.values for p in test_preds_lgb + test_preds_xgb], axis=0),
    columns=FUTURE_TARGET_COLS, index=X_test.index
).clip(lower=0)
sub = build_submission(test_ens, X_test, inference_ts, test_model_df, bias_factor=bias_factor)
sub_path = f"submission_{TRACK}_{EXPERIMENT_NAME}.csv"
sub.to_csv(sub_path, index=False)
print(f"  Saved: {sub_path}  ({len(sub)} rows)")
sys.stdout.flush()
