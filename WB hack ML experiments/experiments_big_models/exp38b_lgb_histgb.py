"""Experiment 38b: Resume exp38 — load saved LGB models + add HistGB.

exp38 failed on HistGB cardinality (route_id has 1000 categories > HistGB limit 255).
Fix: encode cat features as ordinal int (cat.codes) without declaring as categorical.

Loads 7 pre-saved LGB models from exp38, trains 3 HistGB seeds, then ensembles all 10.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib
import json
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, ".")

from config import FUTURE_TARGET_COLS
from data import load_data, create_future_targets, build_feature_cols, split_data, encode_categoricals
from features import make_features
from metrics import WapePlusRbias
from train import predict_steps, build_submission

EXPERIMENT_NAME = "exp38b_lgb_histgb"
LGB_SEEDS = [42, 123, 456, 789, 1234, 2024, 7]
HGB_SEEDS = [42, 123, 456]
ALPHA = 0.543

BASE_HGB = dict(
    loss="quantile",
    quantile=ALPHA,
    max_iter=1000,
    learning_rate=0.05,
    max_leaf_nodes=127,
    min_samples_leaf=20,
    l2_regularization=1.0,
    max_bins=255,
    early_stopping=False,
)

print("=" * 60)
print(f"EXPERIMENT: {EXPERIMENT_NAME}")
print(f"  Load 7 LGB from exp38 + train 3 HistGB (ordinal cat encoding)")
print(f"  alpha={ALPHA} | 154 features | per-step calib")
print("=" * 60)
sys.stdout.flush()


def add_winning_features(df):
    for w in [4, 8, 48]:
        std_col, mean_col = f"target_roll_std_{w}", f"target_roll_mean_{w}"
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


def train_hgb_models(X_fit, y_fit, cat_features, seed, hgb_params):
    """HistGB with ordinal-encoded cat features (works around cardinality>255 limit)."""
    X_fit_hgb = X_fit.copy()
    for col in cat_features:
        if col in X_fit_hgb.columns:
            X_fit_hgb[col] = X_fit_hgb[col].cat.codes.astype(np.float32)

    models = {}
    for step_col in FUTURE_TARGET_COLS:
        params = {**hgb_params, "random_state": seed}
        m = HistGradientBoostingRegressor(**params)
        m.fit(X_fit_hgb, y_fit[step_col].clip(lower=0))
        models[step_col] = m
        print(f"    {step_col:20s}  iters={m.n_iter_}")
        sys.stdout.flush()
    return models


def predict_hgb(models, X, cat_features):
    X_hgb = X.copy()
    for col in cat_features:
        if col in X_hgb.columns:
            X_hgb[col] = X_hgb[col].cat.codes.astype(np.float32)
    preds = {col: np.clip(models[col].predict(X_hgb), 0, None) for col in FUTURE_TARGET_COLS}
    return pd.DataFrame(preds, index=X.index)


# -- Data & features ----------------------------------------------------------
print("Loading data...")
sys.stdout.flush()
train_df, test_df = load_data()
print(f"Routes: {train_df['route_id'].nunique()} | Train rows: {len(train_df)}")
sys.stdout.flush()

print("Building features...")
sys.stdout.flush()
train_df = make_features(train_df, extended=True)
train_df = add_winning_features(train_df)
train_df = create_future_targets(train_df)
feature_cols = build_feature_cols(train_df)
print(f"Feature count: {len(feature_cols)}")
sys.stdout.flush()

X_fit, y_fit, X_valid, y_valid, X_test, inference_ts, test_model_df = split_data(train_df, feature_cols)
X_fit, X_valid, X_test, cat_features = encode_categoricals(X_fit, X_valid, X_test, feature_cols)
print(f"Categorical features: {cat_features}")
sys.stdout.flush()

metric = WapePlusRbias()
exp38_dir = Path("models") / "exp38_lgb_histgb"
models_dir = Path("models") / EXPERIMENT_NAME
models_dir.mkdir(parents=True, exist_ok=True)

valid_preds_list = []
test_preds_list = []

# -- Load pre-trained LGB models from exp38 -----------------------------------
print(f"\n=== Loading {len(LGB_SEEDS)} LGB models from exp38 ===")
sys.stdout.flush()
for seed in LGB_SEEDS:
    path = exp38_dir / f"lgb_seed{seed}.pkl"
    print(f"  Loading {path}...")
    sys.stdout.flush()
    models = joblib.load(path)
    vp = predict_steps(models, X_valid)
    tp = predict_steps(models, X_test)
    w, r, t = metric.calculate_components(y_valid, vp)
    print(f"  LGB seed={seed}: WAPE={w:.4f} RBias={r:.4f} Total={t:.4f}")
    sys.stdout.flush()
    valid_preds_list.append(vp)
    test_preds_list.append(tp)

# -- Train HistGB models ------------------------------------------------------
print(f"\n=== Training {len(HGB_SEEDS)} HistGB seeds (alpha={ALPHA}, ordinal cats) ===")
sys.stdout.flush()
for seed in HGB_SEEDS:
    t0 = datetime.now()
    print(f"\n--- HistGB seed={seed} [{t0.strftime('%H:%M:%S')}] ---")
    sys.stdout.flush()
    models = train_hgb_models(X_fit, y_fit, cat_features, seed, BASE_HGB)
    vp = predict_hgb(models, X_valid, cat_features)
    tp = predict_hgb(models, X_test, cat_features)
    w, r, t = metric.calculate_components(y_valid, vp)
    elapsed = (datetime.now() - t0).seconds
    print(f"  HistGB seed={seed}: WAPE={w:.4f} RBias={r:.4f} Total={t:.4f}  [{elapsed}s]")
    sys.stdout.flush()
    valid_preds_list.append(vp)
    test_preds_list.append(tp)
    joblib.dump(models, models_dir / f"hgb_seed{seed}.pkl")

# -- Ensemble -----------------------------------------------------------------
print(f"\n[Ensemble] Averaging {len(valid_preds_list)} predictions (7 LGB + 3 HistGB)...")
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
print(f"[Raw]        WAPE={w_raw:.4f} RBias={r_raw:.4f} Total={t_raw:.4f}")
sys.stdout.flush()

# Per-step calibration
valid_cal = valid_ens.copy()
test_cal = test_ens.copy()
factors = {}
for c in FUTURE_TARGET_COLS:
    f = float(y_valid[c].sum()) / max(float(valid_ens[c].sum()), 1e-9)
    valid_cal[c] = (valid_ens[c] * f).clip(lower=0)
    test_cal[c] = (test_ens[c] * f).clip(lower=0)
    factors[c] = round(f, 4)

w_cal, r_cal, t_cal = metric.calculate_components(y_valid, valid_cal)
print(f"[Calibrated] WAPE={w_cal:.4f} RBias={r_cal:.4f} Total={t_cal:.4f}")
print(f"[Cal factors]: {list(factors.values())}")
sys.stdout.flush()

# -- Submission ---------------------------------------------------------------
sub = build_submission(test_cal, X_test, inference_ts, test_df)
sub_path = f"submission_team_{EXPERIMENT_NAME}.csv"
sub.to_csv(sub_path, index=False)
print(f"Saved: {sub_path}  ({len(sub)} rows)")
sys.stdout.flush()

# -- Log ----------------------------------------------------------------------
exp_path = Path("experiments.json")
experiments = json.loads(exp_path.read_text(encoding="utf-8"))
experiments.append({
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "name": EXPERIMENT_NAME,
    "params": {
        "lgb_seeds": LGB_SEEDS, "hgb_seeds": HGB_SEEDS,
        "alpha": ALPHA, "num_leaves": 127,
        "n_models": len(valid_preds_list),
        "features": "extended+rolling_cv+lag_ratios (154)",
        "calibration": "per_step",
    },
    "wape": round(w_cal, 6),
    "rbias": round(r_cal, 6),
    "total": round(t_cal, 6),
    "note": (
        f"7 LGB (loaded from exp38) + 3 HistGB ordinal, alpha={ALPHA}. "
        f"Raw: WAPE={w_raw:.4f} RBias={r_raw:.4f}. Cal: total={t_cal:.4f}"
    ),
})
exp_path.write_text(json.dumps(experiments, indent=2, ensure_ascii=False), encoding="utf-8")
print("Logged to experiments.json")

print("\n" + "=" * 60)
print(f"EXPERIMENT {EXPERIMENT_NAME} COMPLETE")
print(f"  Raw:         WAPE={w_raw:.4f}  RBias={r_raw:.4f}  Total={t_raw:.4f}")
print(f"  Calibrated:  WAPE={w_cal:.4f}  RBias={r_cal:.4f}  Total={t_cal:.4f}")
print("=" * 60)
