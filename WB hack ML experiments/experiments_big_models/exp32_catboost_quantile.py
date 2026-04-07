"""Experiment 32: CatBoost Quantile ensemble.

Same feature set as exp_best_features_full (extended + rolling_cv + lag_ratios = 154 feats),
but using CatBoost Quantile(alpha=0.55) instead of LightGBM.
  - 5 seeds: [42, 123, 456, 789, 1234]
  - depth=8, 5000 iterations, lr=0.05, early stopping=100
  - Per-step bias calibration

Hypotheses:
  - CatBoost handles categoricals natively — may generalise better
  - Diversity vs LGB ensemble may help in a mixed ensemble later
"""

import numpy as np
import pandas as pd
import catboost as cb
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
from train import build_submission

EXPERIMENT_NAME = "exp32_catboost_quantile"
SEEDS = [42, 123, 456, 789, 1234]
ALPHA = 0.55

CB_BASE = dict(
    iterations=5000,
    learning_rate=0.05,
    depth=8,
    loss_function=f"Quantile:alpha={ALPHA}",
    eval_metric=f"Quantile:alpha={ALPHA}",
    verbose=False,
    thread_count=-1,
    od_type="Iter",
    od_wait=100,
    use_best_model=True,
)

print("=" * 60)
print(f"EXPERIMENT: {EXPERIMENT_NAME}")
print(f"  100% routes | {len(SEEDS)} seeds | CatBoost Quantile alpha={ALPHA}")
print(f"  +7 winning features (rolling_cv + lag_ratios) | per-step calib")
print("=" * 60)
sys.stdout.flush()


def add_winning_features(df):
    """Add the 7 winning features from screening: rolling_cv + lag_ratios."""
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


def train_cb_models(X_fit, y_fit, X_valid, y_valid, cat_features, seed):
    """Train one CatBoostRegressor per forecast step."""
    params = {**CB_BASE, "random_seed": seed}
    # CatBoost needs string categoricals
    X_fit_cb = X_fit.copy()
    X_valid_cb = X_valid.copy()
    for col in cat_features:
        if col in X_fit_cb.columns:
            X_fit_cb[col] = X_fit_cb[col].astype(str)
            X_valid_cb[col] = X_valid_cb[col].astype(str)

    # column indices for categorical features
    cat_idx = [list(X_fit_cb.columns).index(c) for c in cat_features if c in X_fit_cb.columns]

    models = {}
    for step_col in FUTURE_TARGET_COLS:
        train_pool = cb.Pool(X_fit_cb, y_fit[step_col].clip(lower=0), cat_features=cat_idx)
        valid_pool = cb.Pool(X_valid_cb, y_valid[step_col].clip(lower=0), cat_features=cat_idx)
        m = cb.CatBoostRegressor(**params)
        m.fit(train_pool, eval_set=valid_pool)
        models[step_col] = m
        metric_key = f"Quantile:alpha={ALPHA}"
        best_score = m.best_score_["validation"].get(metric_key, float("nan"))
        print(f"    {step_col:20s}  iter={m.best_iteration_:4d}  score={best_score:.4f}")
        sys.stdout.flush()
    return models


def predict_cb(models, X, cat_features):
    """Predict with CatBoost models."""
    X_cb = X.copy()
    for col in cat_features:
        if col in X_cb.columns:
            X_cb[col] = X_cb[col].astype(str)
    preds = {col: np.clip(models[col].predict(X_cb), 0, None) for col in FUTURE_TARGET_COLS}
    return pd.DataFrame(preds, index=X.index)


# ── Load & feature engineering ────────────────────────────────────────────────
print("Loading data...")
sys.stdout.flush()
train_df, test_df = load_data()
all_routes = train_df["route_id"].unique()
print(f"Routes: {len(all_routes)} (full) | Train rows: {len(train_df)}")
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
print(f"Categorical features: {cat_features}")
sys.stdout.flush()

metric = WapePlusRbias()

models_dir = Path("models") / EXPERIMENT_NAME
models_dir.mkdir(parents=True, exist_ok=True)
print(f"Models will be saved to: {models_dir}")
sys.stdout.flush()

valid_preds_list = []
test_preds_list = []

# ── Train per seed ────────────────────────────────────────────────────────────
for seed in SEEDS:
    t0 = datetime.now()
    print(f"\n--- CatBoost Quantile alpha={ALPHA} seed={seed} [{t0.strftime('%H:%M:%S')}] ---")
    sys.stdout.flush()
    models = train_cb_models(X_fit, y_fit, X_valid, y_valid, cat_features, seed)
    vp = predict_cb(models, X_valid, cat_features)
    tp = predict_cb(models, X_test, cat_features)
    w, r, t = metric.calculate_components(y_valid, vp)
    elapsed = (datetime.now() - t0).seconds
    print(f"  seed={seed}: WAPE={w:.4f} RBias={r:.4f} Total={t:.4f}  [{elapsed}s]")
    sys.stdout.flush()
    valid_preds_list.append(vp)
    test_preds_list.append(tp)
    path = models_dir / f"cb_a055_seed{seed}.pkl"
    joblib.dump(models, path)
    print(f"  Saved: {path}")
    sys.stdout.flush()

# ── Ensemble ──────────────────────────────────────────────────────────────────
print(f"\n[Ensemble] Averaging {len(SEEDS)} CatBoost predictions...")
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
print(f"[Ensemble raw] WAPE={w_raw:.4f} RBias={r_raw:.4f} Total={t_raw:.4f}")
sys.stdout.flush()

# ── Per-step bias calibration ─────────────────────────────────────────────────
print("\n[Per-step calibration]")
sys.stdout.flush()
valid_cal = valid_ens.copy()
test_cal = test_ens.copy()
for c in FUTURE_TARGET_COLS:
    f = float(y_valid[c].sum()) / max(float(valid_ens[c].sum()), 1e-9)
    valid_cal[c] = (valid_ens[c] * f).clip(lower=0)
    test_cal[c] = (test_ens[c] * f).clip(lower=0)

w_cal, r_cal, t_cal = metric.calculate_components(y_valid, valid_cal)
print(f"[After calibration] WAPE={w_cal:.4f} RBias={r_cal:.4f} Total={t_cal:.4f}")
sys.stdout.flush()

# ── Submission ────────────────────────────────────────────────────────────────
print("\n[Submission] Building...")
sys.stdout.flush()
sub = build_submission(test_cal, X_test, inference_ts, test_df)
sub_path = f"submission_team_{EXPERIMENT_NAME}.csv"
sub.to_csv(sub_path, index=False)
print(f"Saved submission: {sub_path}  ({len(sub)} rows)")
sys.stdout.flush()

# ── Log to experiments.json ───────────────────────────────────────────────────
exp_path = Path("experiments.json")
try:
    experiments = json.loads(exp_path.read_text(encoding="utf-8"))
except Exception:
    experiments = []

record = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "name": EXPERIMENT_NAME,
    "params": {
        "sample_frac": 1.0,
        "seeds": SEEDS,
        "alpha": ALPHA,
        "model": "CatBoost Quantile",
        "iterations": 5000,
        "depth": 8,
        "features": "extended+rolling_cv+lag_ratios (154)",
        "calibration": "per_step",
    },
    "wape": round(w_cal, 6),
    "rbias": round(r_cal, 6),
    "total": round(t_cal, 6),
    "note": (
        f"CatBoost Quantile alpha={ALPHA}, 5 seeds. "
        f"Raw: WAPE={w_raw:.4f} RBias={r_raw:.4f} Total={t_raw:.4f}; "
        f"Calibrated: WAPE={w_cal:.4f} RBias={r_cal:.4f} Total={t_cal:.4f}"
    ),
}
experiments.append(record)
exp_path.write_text(json.dumps(experiments, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nLogged to {exp_path}")

print("\n" + "=" * 60)
print(f"EXPERIMENT {EXPERIMENT_NAME} COMPLETE")
print(f"  Calibrated:  WAPE={w_cal:.4f}  RBias={r_cal:.4f}  Total={t_cal:.4f}")
print("=" * 60)
