"""Generate submission for exp28 from saved models."""

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from pathlib import Path
import sys
sys.path.insert(0, ".")

from config import TRACK, FUTURE_TARGET_COLS
from data import load_data, create_future_targets, build_feature_cols, split_data, encode_categoricals
from features import make_features
from train import predict_steps, build_submission

EXPERIMENT_NAME = "exp28_baseline_full"
SEEDS = [42, 123, 456]
MODELS_DIR = Path("models") / EXPERIMENT_NAME

print("Loading data...")
sys.stdout.flush()
train_df, test_df = load_data()  # test_df has 'id' column — needed for submission

print("Building features...")
sys.stdout.flush()
train_df = make_features(train_df, extended=True)
train_df = create_future_targets(train_df)
feature_cols = build_feature_cols(train_df)
print(f"Feature count: {len(feature_cols)}")
sys.stdout.flush()

X_fit, y_fit, X_valid, y_valid, X_test, inference_ts, _ = split_data(train_df, feature_cols)
X_fit, X_valid, X_test, cat_features = encode_categoricals(X_fit, X_valid, X_test, feature_cols)

X_test_xgb = X_test.copy()
for col in cat_features:
    X_test_xgb[col] = X_test_xgb[col].cat.codes.astype(np.int32)

test_preds = []

print("\n--- Loading LGB models and predicting ---")
sys.stdout.flush()
for seed in SEEDS:
    path = MODELS_DIR / f"lgb_seed{seed}.pkl"
    models = joblib.load(path)
    tp = predict_steps(models, X_test)
    test_preds.append(tp)
    print(f"  LGB seed={seed}: loaded from {path}")
    sys.stdout.flush()

print("\n--- Loading XGB models and predicting ---")
sys.stdout.flush()
for seed in SEEDS:
    xgb_dir = MODELS_DIR / f"xgb_seed{seed}"
    preds = {}
    for sc in FUTURE_TARGET_COLS:
        m = XGBRegressor()
        m.load_model(str(xgb_dir / f"{sc}.ubj"))
        preds[sc] = np.clip(m.predict(X_test_xgb), 0, None)
    tp = pd.DataFrame(preds, index=X_test.index)
    test_preds.append(tp)
    print(f"  XGB seed={seed}: loaded from {xgb_dir}/")
    sys.stdout.flush()

test_ens = pd.DataFrame(
    np.mean([p.values for p in test_preds], axis=0),
    columns=FUTURE_TARGET_COLS, index=X_test.index
).clip(lower=0)

# bias_factor = 1.0 (per-step correction was applied at validation level;
# for test we pass factor=1.0 and let build_submission handle the scaling)
sub = build_submission(test_ens, X_test, inference_ts, test_df, bias_factor=1.0)
sub_path = f"submission_{TRACK}_{EXPERIMENT_NAME}.csv"
sub.to_csv(sub_path, index=False)
print(f"\nSaved: {sub_path}  ({len(sub)} rows)")
print("Done.")
