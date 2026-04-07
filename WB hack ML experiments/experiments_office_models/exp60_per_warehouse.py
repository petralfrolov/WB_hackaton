"""Experiment 60: Per-warehouse (office_from_id) models.

Hypothesis: training separate LightGBM models for each source warehouse
captures warehouse-specific shipping patterns better than a single global model.
53 unique warehouses (1–64 routes each, avg 19 routes).

Design:
  - 3 seeds per warehouse  →  53 warehouses × 3 seeds × 10 steps = 1590 model fits
  - Features: extended + rolling_cv + lag_ratios (154 features)
  - Anti-overfitting params (for smaller per-warehouse datasets):
      num_leaves=63, min_child_samples=30, colsample_bytree=0.7, subsample=0.8
      reg_lambda=2.0, n_estimators=2000 + early stopping 100
  - alpha=0.55 (quantile regression)
  - train_days=21 (same window as best global models)
  - Per-step bias calibration (global, on combined validation predictions)
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from config import FUTURE_TARGET_COLS
from data import (build_feature_cols, create_future_targets, encode_categoricals,
                  load_data, split_data)
from features import make_features
from metrics import WapePlusRbias
from train import build_submission, predict_steps

EXPERIMENT_NAME = "exp60_per_warehouse"
ALPHA = 0.55
SEEDS = [42, 123, 456]
TRAIN_DAYS = 21

# Anti-overfitting parameters for per-warehouse (smaller dataset) models
BASE_PARAMS = dict(
    objective="quantile",
    alpha=ALPHA,
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=2.0,
    n_jobs=-1,
    verbose=-1,
)

print("=" * 60)
print(f"EXPERIMENT: {EXPERIMENT_NAME}")
print(f"  Per-warehouse LightGBM, {len(SEEDS)} seeds each warehouse")
print(f"  num_leaves=63, min_child_samples=30, cs=0.7, ss=0.8")
print("=" * 60)
sys.stdout.flush()


# ── Winning extra features ─────────────────────────────────────────────────────
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


def train_warehouse_models(Xf, yf, Xv, yv, params):
    """Train one LGB per forecast step for a given warehouse subset."""
    models = {}
    for step_col in FUTURE_TARGET_COLS:
        m = lgb.LGBMRegressor(**params)
        m.fit(Xf, yf[step_col],
              eval_set=[(Xv, yv[step_col])],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(0)])
        models[step_col] = m
    return models


# ── 1. Load + build features ───────────────────────────────────────────────────
print("\n[1] Loading data and building features...")
sys.stdout.flush()
train_df_raw, test_df = load_data()
train_df = make_features(train_df_raw, extended=True)
train_df = add_winning_features(train_df)
train_df = create_future_targets(train_df)
feature_cols = build_feature_cols(train_df)
print(f"Feature count: {len(feature_cols)}")
sys.stdout.flush()

# ── 2. Global split + encode ───────────────────────────────────────────────────
print("\n[2] Splitting data (train_days=21)...")
sys.stdout.flush()
X_fit, y_fit, X_valid, y_valid, X_test, inference_ts, _ = split_data(
    train_df, feature_cols, train_days=TRAIN_DAYS
)
X_fit, X_valid, X_test, _ = encode_categoricals(X_fit, X_valid, X_test, feature_cols)
print(f"Fit={len(X_fit):,}  Valid={len(X_valid):,}  Test={len(X_test):,}")
sys.stdout.flush()

# ── 3. Per-warehouse training ──────────────────────────────────────────────────
# office_from_id is Categorical after encode_categoricals; convert to str for masking
fit_wh   = X_fit["office_from_id"].astype(str)
valid_wh = X_valid["office_from_id"].astype(str)
test_wh  = X_test["office_from_id"].astype(str)

warehouses = sorted(test_wh.unique())
print(f"\n[3] Training per-warehouse models: {len(warehouses)} warehouses x {len(SEEDS)} seeds")
sys.stdout.flush()

models_dir = Path("models") / EXPERIMENT_NAME
models_dir.mkdir(parents=True, exist_ok=True)

# Collect warehouse validation + test predictions (keyed by original DataFrame index)
valid_pred_parts = []   # list of DataFrames with original index
test_pred_parts  = []   # list of DataFrames with original index
metric = WapePlusRbias()

for wh_idx, wh in enumerate(warehouses):
    mask_fit   = fit_wh   == wh
    mask_valid = valid_wh == wh
    mask_test  = test_wh  == wh

    Xf = X_fit[mask_fit];    yf = y_fit[mask_fit]
    Xv = X_valid[mask_valid]; yv = y_valid[mask_valid]
    Xt = X_test[mask_test]

    n_fit_routes = int(mask_fit.sum())
    n_routes_test = int(mask_test.sum())
    t_wh = datetime.now()
    print(f"\n[WH {wh_idx+1:2d}/{len(warehouses)}] office_from_id={wh} "
          f"| fit={n_fit_routes} valid={int(mask_valid.sum())} test={n_routes_test}")
    sys.stdout.flush()

    if n_fit_routes < 10:
        # Too few fit samples for per-warehouse model — use ultra-simple params
        wh_params_base = {**BASE_PARAMS, "num_leaves": 15, "min_child_samples": 10}
        print(f"  WARNING: very few fit rows ({n_fit_routes}), using simplified params (leaves=15)")
    else:
        wh_params_base = BASE_PARAMS.copy()

    # Collect per-seed accumulated predictions for this warehouse
    vp_list_wh = []  # each entry: DataFrame[FUTURE_TARGET_COLS] with index from X_valid
    tp_list_wh = []  # each entry: DataFrame[FUTURE_TARGET_COLS] with index from X_test

    for seed in SEEDS:
        params = {**wh_params_base, "random_state": seed}
        models = train_warehouse_models(Xf, yf, Xv, yv, params)

        vp = predict_steps(models, Xv)   # index = X_valid subset index
        tp = predict_steps(models, Xt)   # index = X_test  subset index
        vp_list_wh.append(vp)
        tp_list_wh.append(tp)

        best_iters = {col: models[col].best_iteration_ for col in FUTURE_TARGET_COLS}
        avg_iter = int(np.mean(list(best_iters.values())))
        w, r, t = metric.calculate_components(yv, vp.clip(lower=0))
        print(f"  seed={seed}: avg_iter={avg_iter:4d}  WAPE={w:.4f} RBias={r:.4f} Total={t:.4f}")
        sys.stdout.flush()

        joblib.dump(models, models_dir / f"wh{wh}_seed{seed}.pkl")

    # Average seeds for this warehouse
    vp_wh = pd.DataFrame(
        np.mean([p.values for p in vp_list_wh], axis=0),
        columns=FUTURE_TARGET_COLS, index=Xv.index
    ).clip(lower=0)
    tp_wh = pd.DataFrame(
        np.mean([p.values for p in tp_list_wh], axis=0),
        columns=FUTURE_TARGET_COLS, index=Xt.index
    ).clip(lower=0)

    w_wh, r_wh, t_wh_val = metric.calculate_components(yv, vp_wh)
    elapsed = (datetime.now() - t_wh).seconds
    print(f"  WH {wh} ensemble ({len(SEEDS)} seeds): WAPE={w_wh:.4f} RBias={r_wh:.4f} Total={t_wh_val:.4f}  [{elapsed}s]")
    sys.stdout.flush()

    valid_pred_parts.append(vp_wh)
    test_pred_parts.append(tp_wh)

# ── 4. Assemble global predictions ────────────────────────────────────────────
print("\n[4] Assembling per-warehouse predictions...")
sys.stdout.flush()

# Sort by original index to align with y_valid ordering
valid_ens = pd.concat(valid_pred_parts).sort_index()
test_ens  = pd.concat(test_pred_parts).sort_index()

# Align y_valid to the same index (some warehouses might be missing if not in fit)
y_valid_aligned = y_valid.loc[valid_ens.index]

w_raw, r_raw, t_raw = metric.calculate_components(y_valid_aligned, valid_ens)
print(f"[Raw]  WAPE={w_raw:.4f}  RBias={r_raw:.4f}  Total={t_raw:.4f}")
print(f"       (valid coverage: {len(valid_ens):,} / {len(y_valid):,} rows)")
sys.stdout.flush()

# ── 5. Per-step bias calibration ──────────────────────────────────────────────
print("\n[5] Per-step calibration...")
factors = {}
valid_cal = valid_ens.copy()
test_cal  = test_ens.copy()
for col in FUTURE_TARGET_COLS:
    f = float(y_valid_aligned[col].sum()) / max(float(valid_ens[col].sum()), 1e-9)
    valid_cal[col] = (valid_ens[col] * f).clip(lower=0)
    test_cal[col]  = (test_ens[col]  * f).clip(lower=0)
    factors[col] = round(f, 4)

w_cal, r_cal, t_cal = metric.calculate_components(y_valid_aligned, valid_cal)
print(f"[Cal]  WAPE={w_cal:.4f}  RBias={r_cal:.4f}  Total={t_cal:.4f}")
print(f"[Cal factors]: {list(factors.values())}")
sys.stdout.flush()

# ── 6. Build and save submission ──────────────────────────────────────────────
sub = build_submission(test_cal, X_test, inference_ts, test_df)
sub_path = f"submission_team_{EXPERIMENT_NAME}.csv"
sub.to_csv(sub_path, index=False)
print(f"\nSaved submission: {sub_path}  ({len(sub)} rows)")

# ── 7. Log to experiments.json ────────────────────────────────────────────────
exp_path = Path("experiments.json")
exps = json.loads(exp_path.read_text(encoding="utf-8"))
exps.append({
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "name": EXPERIMENT_NAME,
    "params": {
        "n_warehouses": len(warehouses),
        "seeds_per_warehouse": SEEDS,
        "alpha": ALPHA,
        "num_leaves": 63,
        "min_child_samples": 30,
        "colsample_bytree": 0.7,
        "subsample": 0.8,
        "reg_lambda": 2.0,
        "n_models_total": len(warehouses) * len(SEEDS),
        "features": "extended+rolling_cv+lag_ratios (154)",
        "train_days": TRAIN_DAYS,
        "calibration": "per_step",
    },
    "wape": round(w_cal, 6),
    "rbias": round(r_cal, 6),
    "total": round(t_cal, 6),
    "note": (
        f"Per-warehouse: {len(warehouses)} warehouses × {len(SEEDS)} seeds, "
        f"leaves=63 min_child=30 cs=0.7 ss=0.8 l2=2.0. "
        f"Raw: WAPE={w_raw:.4f} RBias={r_raw:.4f} Total={t_raw:.4f}. "
        f"Cal: total={t_cal:.4f}. "
        f"Cal factors: {list(factors.values())}"
    ),
})
exp_path.write_text(json.dumps(exps, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Logged to experiments.json")

print("\n" + "=" * 60)
print(f"EXPERIMENT {EXPERIMENT_NAME} COMPLETE")
print(f"  Raw: WAPE={w_raw:.4f}  RBias={r_raw:.4f}  Total={t_raw:.4f}")
print(f"  Cal: WAPE={w_cal:.4f}  RBias={r_cal:.4f}  Total={t_cal:.4f}")
print("=" * 60)
