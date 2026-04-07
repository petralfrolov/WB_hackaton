"""Build submission for exp_best_features_full from saved models."""
import numpy as np, pandas as pd, joblib
from pathlib import Path
import sys
sys.path.insert(0, ".")
from config import TRACK, FUTURE_TARGET_COLS
from data import load_data, create_future_targets, build_feature_cols, split_data, encode_categoricals
from features import make_features
from train import predict_steps, build_submission

train_df, test_df = load_data()

def add_winning_features(df):
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

train_df = make_features(train_df, extended=True)
train_df = add_winning_features(train_df)
train_df = create_future_targets(train_df)
feature_cols = build_feature_cols(train_df)
X_fit, y_fit, X_valid, y_valid, X_test, inference_ts, test_model_df = split_data(train_df, feature_cols)
X_fit, X_valid, X_test, cat_features = encode_categoricals(X_fit, X_valid, X_test, feature_cols)

models_dir = Path("models/exp_best_features_full")
SEEDS = [42, 123, 456, 789, 1234]
test_preds, valid_preds = [], []

for alpha_str in ["a052", "a055"]:
    for seed in SEEDS:
        path = models_dir / f"lgb_{alpha_str}_seed{seed}.pkl"
        models = joblib.load(path)
        test_preds.append(predict_steps(models, X_test))
        valid_preds.append(predict_steps(models, X_valid))
        print(f"  Loaded: {path}")

test_ens = pd.DataFrame(
    np.mean([p.values for p in test_preds], axis=0),
    columns=FUTURE_TARGET_COLS, index=X_test.index,
).clip(lower=0)
valid_ens = pd.DataFrame(
    np.mean([p.values for p in valid_preds], axis=0),
    columns=FUTURE_TARGET_COLS, index=X_valid.index,
).clip(lower=0)

test_corrected = test_ens.copy()
for col in FUTURE_TARGET_COLS:
    f = float(y_valid[col].sum()) / max(float(valid_ens[col].sum()), 1e-9)
    test_corrected[col] = (test_ens[col] * f).clip(lower=0)

sub = build_submission(test_corrected, X_test, inference_ts, test_df, bias_factor=1.0)
sub_path = f"submission_{TRACK}_exp_best_features_full.csv"
sub.to_csv(sub_path, index=False)
print(f"Saved: {sub_path}  ({len(sub)} rows)")
