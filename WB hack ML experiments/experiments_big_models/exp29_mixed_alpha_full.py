"""Experiment 29: Exp24 variant E on 100% data — no XGBoost.

Best config from exp24 (20%-routes): mixed alpha (0.52 + 0.55), 10 LGB models,
per-step bias calibration. Reproduces variant E on the full dataset.

Pure LGB only — XGBoost dropped due to poor public-LB generalization (exp28: 0.289 vs exp19: 0.253).

Setup:
- 5 × LGB alpha=0.52, seeds [42, 123, 456, 789, 1234]
- 5 × LGB alpha=0.55, seeds [42, 123, 456, 789, 1234]
- Simple average of all 10 predictions
- Per-step bias calibration (factor computed on validation, applied to test)
- extended features + 12-pair deconvolution
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
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

EXPERIMENT_NAME = "exp29_mixed_alpha_full"
SEEDS = [42, 123, 456, 789, 1234]

BASE_LGB = dict(
    n_estimators=5000, learning_rate=0.05, num_leaves=127,
    min_child_samples=20, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    n_jobs=-1, verbose=-1,
)

print("=" * 60)
print(f"EXPERIMENT: {EXPERIMENT_NAME}")
print(f"  100% routes | 10 LGB models | alpha=0.52+0.55 | per-step calib")
print("=" * 60)
sys.stdout.flush()

print("Loading data...")
sys.stdout.flush()
train_df, test_df = load_data()
all_routes = train_df["route_id"].unique()
print(f"Routes: {len(all_routes)} (full) | Train rows: {len(train_df)}")
sys.stdout.flush()

print("Building features (extended + 12-pair deconvolution)...")
sys.stdout.flush()
train_df = make_features(train_df, extended=True)
train_df = create_future_targets(train_df)
feature_cols = build_feature_cols(train_df)
print(f"Feature count: {len(feature_cols)}")
sys.stdout.flush()

X_fit, y_fit, X_valid, y_valid, X_test, inference_ts, test_model_df = split_data(train_df, feature_cols)
X_fit, X_valid, X_test, cat_features = encode_categoricals(X_fit, X_valid, X_test, feature_cols)

metric = WapePlusRbias()

models_dir = Path("models") / EXPERIMENT_NAME
models_dir.mkdir(parents=True, exist_ok=True)
print(f"Models will be saved to: {models_dir}")
sys.stdout.flush()

valid_preds_list = []
test_preds_list = []

# ── alpha=0.52 models ──────────────────────────────────────────────────────────
print("\n--- LGB alpha=0.52 ---")
sys.stdout.flush()
for seed in SEEDS:
    t0 = datetime.now()
    print(f"  seed={seed} started at {t0.strftime('%H:%M:%S')} ...")
    sys.stdout.flush()
    p = {**BASE_LGB, "objective": "quantile", "alpha": 0.52, "random_state": seed}
    models = train_lgb_models(X_fit, y_fit, X_valid, y_valid, lgb_params=p)
    vp = predict_steps(models, X_valid)
    tp = predict_steps(models, X_test)
    w, r, t = metric.calculate_components(y_valid, vp)
    elapsed = (datetime.now() - t0).seconds
    print(f"  seed={seed} alpha=0.52: WAPE={w:.4f} RBias={r:.4f} Total={t:.4f}  [{elapsed}s]")
    sys.stdout.flush()
    valid_preds_list.append(vp)
    test_preds_list.append(tp)
    path = models_dir / f"lgb_a052_seed{seed}.pkl"
    joblib.dump(models, path)
    print(f"  Saved: {path}")
    sys.stdout.flush()

# ── alpha=0.55 models ──────────────────────────────────────────────────────────
print("\n--- LGB alpha=0.55 ---")
sys.stdout.flush()
for seed in SEEDS:
    t0 = datetime.now()
    print(f"  seed={seed} started at {t0.strftime('%H:%M:%S')} ...")
    sys.stdout.flush()
    p = {**BASE_LGB, "objective": "quantile", "alpha": 0.55, "random_state": seed}
    models = train_lgb_models(X_fit, y_fit, X_valid, y_valid, lgb_params=p)
    vp = predict_steps(models, X_valid)
    tp = predict_steps(models, X_test)
    w, r, t = metric.calculate_components(y_valid, vp)
    elapsed = (datetime.now() - t0).seconds
    print(f"  seed={seed} alpha=0.55: WAPE={w:.4f} RBias={r:.4f} Total={t:.4f}  [{elapsed}s]")
    sys.stdout.flush()
    valid_preds_list.append(vp)
    test_preds_list.append(tp)
    path = models_dir / f"lgb_a055_seed{seed}.pkl"
    joblib.dump(models, path)
    print(f"  Saved: {path}")
    sys.stdout.flush()

# ── Ensemble ───────────────────────────────────────────────────────────────────
print("\n[Ensemble] Averaging 10 predictions...")
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
print(f"[Ensemble raw]  WAPE={w_raw:.4f} RBias={r_raw:.4f} Total={t_raw:.4f}")
sys.stdout.flush()

# ── Per-step calibration ───────────────────────────────────────────────────────
print("\n[Per-step calibration]")
per_step_factors = {}
valid_corrected = valid_ens.copy()
test_corrected = test_ens.copy()

for col in FUTURE_TARGET_COLS:
    true_sum = float(y_valid[col].sum())
    pred_sum = float(valid_ens[col].sum())
    f = true_sum / pred_sum if pred_sum > 0 else 1.0
    per_step_factors[col] = f
    valid_corrected[col] = (valid_ens[col] * f).clip(lower=0)
    test_corrected[col] = (test_ens[col] * f).clip(lower=0)
    print(f"  {col}: factor={f:.4f}")
sys.stdout.flush()

w_c, r_c, t_c = metric.calculate_components(y_valid, valid_corrected)
print(f"\n[Ensemble per-step] WAPE={w_c:.4f} RBias={r_c:.4f} Total={t_c:.4f}")
sys.stdout.flush()

# ── Log ────────────────────────────────────────────────────────────────────────
log_path = Path("experiments.json")
history = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
history.append({
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "name": EXPERIMENT_NAME,
    "params": dict(
        sample_frac=1.0,
        seeds=SEEDS,
        alphas=[0.52, 0.55],
        n_models=10,
        calibration="per_step",
        models="10xLGB",
    ),
    "wape": round(w_c, 6),
    "rbias": round(r_c, 6),
    "total": round(t_c, 6),
    "note": f"Exp24-E on full data. Raw WAPE={w_raw:.4f} -> per-step corrected={t_c:.4f}",
})
log_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
print("\nLogged to experiments.json")
sys.stdout.flush()

# ── Submission ─────────────────────────────────────────────────────────────────
print("\n[Submission] Building test predictions...")
sys.stdout.flush()
# bias_factor=1.0 because per-step correction already applied to test_corrected
sub = build_submission(test_corrected, X_test, inference_ts, test_df, bias_factor=1.0)
sub_path = f"submission_{TRACK}_{EXPERIMENT_NAME}.csv"
sub.to_csv(sub_path, index=False)
print(f"  Saved: {sub_path}  ({len(sub)} rows)")
print(f"\nDONE. VALID total = {t_c:.6f}")
sys.stdout.flush()
