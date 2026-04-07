"""Rebuild exp58 submissions:
 - exp58b_nocal:    raw predictions (no calibration)
 - exp58c_cal51:    exp51 calibration factors applied
"""
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
from config import FUTURE_TARGET_COLS
from data import build_feature_cols, create_future_targets, encode_categoricals, load_data, split_data
from features import make_features
from train import build_submission, predict_steps

# exp51 calibration factors (from exp51_out.txt)
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

print("Loading data + features...")
train_df_raw, test_df = load_data()
train_df = make_features(train_df_raw, extended=True)
train_df = add_winning_features(train_df)
train_df = create_future_targets(train_df)
feature_cols = build_feature_cols(train_df)

X_fit, y_fit, X_valid, y_valid, X_test, inference_ts, _ = split_data(train_df, feature_cols, train_days=21)
X_fit, X_valid, X_test, _ = encode_categoricals(X_fit, X_valid, X_test, feature_cols)
print(f"Test rows: {len(X_test)}")

models_dir = Path("models") / "exp58_full_window"
model_files = sorted(models_dir.glob("lgb_*.pkl"))
print(f"Found {len(model_files)} model files")

tp_list = []
for mf in model_files:
    models = joblib.load(mf)
    tp = predict_steps(models, X_test)
    tp_list.append(tp)
    print(f"  {mf.name}: loaded")

test_ens = pd.DataFrame(
    np.mean([p.values for p in tp_list], axis=0),
    columns=FUTURE_TARGET_COLS, index=X_test.index
).clip(lower=0)

# --- Submission 1: no calibration ---
sub_nocal = build_submission(test_ens, X_test, inference_ts, test_df)
sub_nocal.to_csv("submission_team_exp58b_nocal.csv", index=False)
print(f"\nSaved: submission_team_exp58b_nocal.csv  ({len(sub_nocal)} rows)")

# --- Submission 2: exp51 calibration factors ---
test_cal51 = test_ens.copy()
for col in FUTURE_TARGET_COLS:
    test_cal51[col] = (test_ens[col] * CAL51[col]).clip(lower=0)
sub_cal51 = build_submission(test_cal51, X_test, inference_ts, test_df)
sub_cal51.to_csv("submission_team_exp58c_cal51.csv", index=False)
print(f"Saved: submission_team_exp58c_cal51.csv  ({len(sub_cal51)} rows)")

print("\nDone.")
