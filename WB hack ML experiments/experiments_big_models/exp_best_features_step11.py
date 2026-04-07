"""Train target_step_12 and patch existing exp_best_features_full ensembles.

Why a separate script:
  - The existing 10 pkl files each hold a dict {target_step_1..10: LGBModel}.
  - FORECAST_POINTS=10 covers steps 1-10 (= up to +5h ahead).
  - Step 12 (+6h boundary, 4-6h window) is missing.
  - This script trains step_12 from the identical split/features, then injects
    it into every existing pkl under the key "target_step_12".

After this script:
  - ml_prediction.py will automatically serve pred_4_6h (pred_4_6h_available=true).
"""

import sys
sys.path.insert(0, ".")

import json
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from config import TRACK, TARGET_COL, TRAIN_DAYS, MAX_TRAIN_ROWS, VALID_FRAC, RANDOM_STATE
from data import load_data, build_feature_cols, encode_categoricals
from features import make_features
from metrics import WapePlusRbias

EXPERIMENT_NAME = "exp_best_features_full"
EXTRA_STEP = 11                   # target_step_11
SEEDS = [42, 123, 456, 789, 1234]

BASE_LGB = dict(
    n_estimators=5000, learning_rate=0.05, num_leaves=127,
    min_child_samples=20, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    n_jobs=-1, verbose=-1,
)

# ── Winning features (must match exp_best_features_full) ──────────────────────
def add_winning_features(df):
    for w in [4, 8, 48]:
        sc = f"target_roll_std_{w}"
        mc = f"target_roll_mean_{w}"
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


# ── Custom split that also returns target_step_11 ─────────────────────────────
STEP11_COL = f"target_step_{EXTRA_STEP}"

def split_with_step11(train_df, feature_cols):
    """Identical split logic to split_data() but includes target_step_11 in y."""
    # Build step_11 target via shift
    train_df = train_df.copy()
    g = train_df.groupby("route_id", sort=False)
    train_df[STEP11_COL] = g[TARGET_COL].shift(-EXTRA_STEP)

    supervised_df = train_df.dropna(subset=[STEP11_COL]).copy()
    print(f"  Step-11 supervised rows: {len(supervised_df)}")

    cols_needed = feature_cols + ["timestamp", STEP11_COL]
    model_df = supervised_df[cols_needed].rename(columns={"timestamp": "source_timestamp"})

    ts_max = model_df["source_timestamp"].max()
    ts_start = ts_max - pd.Timedelta(days=TRAIN_DAYS)
    model_df = model_df[model_df["source_timestamp"] >= ts_start].sort_values("source_timestamp")

    split_point = model_df["source_timestamp"].quantile(VALID_FRAC)
    fit_df = model_df[model_df["source_timestamp"] <= split_point].copy()
    valid_df = model_df[model_df["source_timestamp"] > split_point].copy()

    if len(fit_df) > MAX_TRAIN_ROWS:
        fit_df = fit_df.sample(MAX_TRAIN_ROWS, random_state=RANDOM_STATE)

    print(f"  Fit rows: {len(fit_df)}  |  Valid rows: {len(valid_df)}")

    X_fit = fit_df[feature_cols].copy()
    y_fit_11 = fit_df[STEP11_COL].copy()
    X_valid = valid_df[feature_cols].copy()
    y_valid_11 = valid_df[STEP11_COL].copy()

    return X_fit, y_fit_11, X_valid, y_valid_11


# ── Main ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print(f"STEP-11 TRAINING: {EXPERIMENT_NAME}")
print(f"  Trains target_step_11 (+6h) for 10 seeds/alphas")
print(f"  And patches existing pkl files in-place")
print("=" * 60)
sys.stdout.flush()

models_dir = Path("models") / EXPERIMENT_NAME
pkl_files = sorted(models_dir.glob("*.pkl"))
if not pkl_files:
    raise FileNotFoundError(f"No pkl files found in {models_dir}. Run exp_best_features_full.py first.")
print(f"\nFound {len(pkl_files)} existing pkl files to patch:")
for p in pkl_files:
    print(f"  {p.name}")
sys.stdout.flush()

# Check if step_11 is already present in any pkl
already_done = [p for p in pkl_files if STEP11_COL in joblib.load(p)]
if already_done:
    print(f"\nWARNING: {len(already_done)} pkl(s) already contain {STEP11_COL}.")
    print("They will be overwritten with freshly trained models.")
sys.stdout.flush()

print("\nLoading data and building features...")
sys.stdout.flush()
train_df, _ = load_data()

train_df = make_features(train_df, extended=True)
train_df = add_winning_features(train_df)
feature_cols = build_feature_cols(train_df)
print(f"  Feature count: {len(feature_cols)}")
sys.stdout.flush()

X_fit, y_fit_11, X_valid, y_valid_11 = split_with_step11(train_df, feature_cols)

# Encode categoricals using same logic as original experiment
cat_cols = [c for c in feature_cols if c.endswith("_id")]
all_cats_per_col = {
    col: pd.concat([X_fit[col], X_valid[col]]).astype(str).unique()
    for col in cat_cols
}
for col, cats in all_cats_per_col.items():
    dtype = pd.CategoricalDtype(categories=cats)
    X_fit[col] = X_fit[col].astype(str).astype(dtype)
    X_valid[col] = X_valid[col].astype(str).astype(dtype)

metric = WapePlusRbias()

# ── Train one step_11 model per seed×alpha and collect results ────────────────
step11_models: dict[str, lgb.LGBMRegressor] = {}   # key = pkl filename stem

for alpha, alpha_tag in [(0.52, "a052"), (0.55, "a055")]:
    print(f"\n--- LGB alpha={alpha} ---")
    sys.stdout.flush()
    for seed in SEEDS:
        t0 = datetime.now()
        pkl_name = f"lgb_{alpha_tag}_seed{seed}.pkl"
        print(f"  seed={seed} started at {t0.strftime('%H:%M:%S')} ...", end=" ")
        sys.stdout.flush()

        p = {**BASE_LGB, "objective": "quantile", "alpha": alpha, "random_state": seed}
        m = lgb.LGBMRegressor(**p)
        m.fit(
            X_fit, y_fit_11,
            eval_set=[(X_valid, y_valid_11)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        val_pred = np.clip(m.predict(X_valid), 0, None)
        wape = np.abs(val_pred - y_valid_11).sum() / (y_valid_11.sum() + 1e-8)
        elapsed = (datetime.now() - t0).seconds
        print(f"best_iter={m.best_iteration_:4d}  step11_WAPE={wape:.4f}  [{elapsed}s]")
        sys.stdout.flush()

        step11_models[pkl_name] = m

# ── Patch existing pkl files ──────────────────────────────────────────────────
print("\n[Patching pkl files]")
sys.stdout.flush()
for pkl_path in pkl_files:
    ensemble_dict = joblib.load(pkl_path)
    m11 = step11_models.get(pkl_path.name)
    if m11 is None:
        print(f"  SKIP {pkl_path.name} — no matching step_11 model trained (unexpected)")
        continue
    ensemble_dict[STEP11_COL] = m11
    joblib.dump(ensemble_dict, pkl_path)
    total_steps = [k for k in ensemble_dict if k.startswith("target_step_")]
    print(f"  Patched {pkl_path.name}: now has steps {sorted(total_steps)}")
sys.stdout.flush()

# ── Quick per-step validation summary ────────────────────────────────────────
print("\n[Step-11 ensemble validation]")
sys.stdout.flush()
preds_11 = []
for pkl_path in sorted(pkl_files):
    ens = joblib.load(pkl_path)
    preds_11.append(np.clip(ens[STEP11_COL].predict(X_valid), 0, None))

ens_pred_11 = np.mean(preds_11, axis=0)
wape_ens = np.abs(ens_pred_11 - y_valid_11).sum() / (y_valid_11.sum() + 1e-8)
bias_ens = ens_pred_11.sum() / (y_valid_11.sum() + 1e-8) - 1.0
print(f"  Ensemble step_11: WAPE={wape_ens:.4f}  RBias={bias_ens:+.4f}")

# ── Log ────────────────────────────────────────────────────────────────────────
log_path = Path("experiments.json")
history = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
history.append({
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "name": f"{EXPERIMENT_NAME}_step11",
    "params": dict(seeds=SEEDS, alphas=[0.52, 0.55], step=11),
    "step11_wape_ens": round(wape_ens, 6),
    "step11_rbias_ens": round(bias_ens, 6),
})
log_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
print("  Logged to experiments.json")

print(f"\n{'=' * 60}")
print(f"DONE — {STEP11_COL} added to all {len(pkl_files)} pkl files.")
print(f"  ml_prediction.py will now return pred_4_6h (pred_4_6h_available=true).")
print(f"{'=' * 60}")
