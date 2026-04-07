"""Experiment 17: Multi-seed ensemble of the best config (exp15).

Strategy: train 3 LGB models with different random seeds using the best config
(quantile 0.57 + deconv features + extended + 5000 trees). Average their predictions
before applying bias correction. Reduces variance by ~1/sqrt(3) ≈ 42% with no
additional bias.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, ".")

from config import TRACK, CONFIG, TARGET_COL, FUTURE_TARGET_COLS
from data import load_data, create_future_targets, build_feature_cols, split_data, encode_categoricals
from features import make_features
from metrics import WapePlusRbias
from train import train_lgb_models, predict_steps, compute_bias_factor, build_submission

SEEDS = [42, 123, 456]
EXPERIMENT_NAME = "exp17_multiseed_ensemble"

params_base = {
    "n_estimators": 5000,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "quantile",
    "alpha": 0.57,
    "n_jobs": -1,
    "verbose": -1,
}

print("=" * 70)
print(f"EXPERIMENT: {EXPERIMENT_NAME}")
print("=" * 70)

train_df, test_df = load_data()
train_df = make_features(train_df, extended=True)
train_df = create_future_targets(train_df)

feature_cols = build_feature_cols(train_df)
print(f"Feature count: {len(feature_cols)}")

X_fit, y_fit, X_valid, y_valid, X_test, inference_ts, test_model_df = split_data(train_df, feature_cols)
X_fit, X_valid, X_test, cat_features = encode_categoricals(X_fit, X_valid, X_test, feature_cols)

metric = WapePlusRbias()

valid_preds_list = []
test_preds_list = []

for seed in SEEDS:
    print(f"\n--- Seed {seed} ---")
    params = {**params_base, "random_state": seed}
    models = train_lgb_models(X_fit, y_fit, X_valid, y_valid, lgb_params=params)
    valid_preds_list.append(predict_steps(models, X_valid))
    test_preds_list.append(predict_steps(models, X_test))

# Average predictions across seeds
valid_pred_avg = pd.DataFrame(
    np.mean([p.values for p in valid_preds_list], axis=0),
    columns=FUTURE_TARGET_COLS, index=X_valid.index
).clip(lower=0)

test_pred_avg = pd.DataFrame(
    np.mean([p.values for p in test_preds_list], axis=0),
    columns=FUTURE_TARGET_COLS, index=X_test.index
).clip(lower=0)

print("\n[Evaluation — Ensemble]")
w_val, r_val, t_val = metric.calculate_components(y_valid, valid_pred_avg)
print(f"  VALID — WAPE: {w_val:.4f}  RBias: {r_val:.4f}  Total: {t_val:.4f}")

bias_factor = compute_bias_factor(y_valid, valid_pred_avg)
corrected_valid = (valid_pred_avg * bias_factor).clip(lower=0)
w_vc, r_vc, t_vc = metric.calculate_components(y_valid, corrected_valid)
print(f"  VALID (corrected, factor={bias_factor:.4f}) — WAPE: {w_vc:.4f}  RBias: {r_vc:.4f}  Total: {t_vc:.4f}")

# Submission
sub = build_submission(test_pred_avg, X_test, inference_ts, test_df, bias_factor=bias_factor)
sub_path = f"submission_{TRACK}_{EXPERIMENT_NAME}.csv"
sub.to_csv(sub_path, index=False)
print(f"\nSaved: {sub_path}  ({len(sub)} rows)")

# Log
log_path = Path("experiments.json")
history = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
history.append({
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "name": EXPERIMENT_NAME,
    "params": {"seeds": SEEDS, **{k: v for k, v in params_base.items() if k not in ("n_jobs", "verbose")}},
    "wape": round(w_vc, 6),
    "rbias": round(r_vc, 6),
    "total": round(t_vc, 6),
})
log_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Done. VALID metric = {t_vc:.4f}")
