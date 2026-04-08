"""Train target_step_11 and target_step_12 for exp66_roll_ratio_struct_diversity.

Why a separate script:
  - The existing 10 pkl files each hold a dict {target_step_1..10: LGBModel}.
  - Step 11 (+5.5h) and step 12 (+6h boundary) are missing.
  - This script trains both extra steps respecting exp66's group/hyperparameter
    structure, then injects them into every existing pkl.
  - After patching, it exports ALL steps (1-12) of every seed as SEPARATE
    individual .pkl files into ml_exp66_prediction/models/ so all files are
    small enough to push to GitHub (<100 MB each).

After this script:
  - ml_exp66_prediction/prediction.py will serve pred_4_6h.
  - GitHub-uploadable split pkls live in ml_exp66_prediction/models/.
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

from config import TARGET_COL, TRAIN_DAYS, MAX_TRAIN_ROWS, VALID_FRAC, RANDOM_STATE
from data import load_data, build_feature_cols
from features import make_features
from metrics import WapePlusRbias

EXPERIMENT_NAME = "exp66_roll_ratio_struct_diversity"

# ── Group structure identical to exp66 main script ───────────────────────────
GROUPS = [
    ("A_cs08", [42, 123, 456, 789],  0.8, 0.8),
    ("B_cs05", [1234, 2024, 7],      0.5, 0.7),
    ("C_cs07", [314, 99, 888],       0.7, 0.9),
]
ALPHA = 0.55

BASE_LGB = dict(
    n_estimators=5000, learning_rate=0.05, num_leaves=511,
    min_child_samples=10, subsample_freq=1, reg_alpha=0.1, reg_lambda=1.0,
    n_jobs=-1, verbose=-1,
)

# Build a flat map: seed -> (label, cs, ss) for easy lookup
SEED_TO_GROUP: dict[int, tuple] = {}
for label, seeds, cs, ss in GROUPS:
    for seed in seeds:
        SEED_TO_GROUP[seed] = (label, cs, ss)

# ── Feature helpers (must exactly match exp66 main script) ────────────────────

def add_winning_features(df: pd.DataFrame) -> pd.DataFrame:
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


def add_roll_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    if "target_roll_mean_8" in df.columns and "target_roll_mean_96" in df.columns:
        df["roll_ratio_8_96"] = df["target_roll_mean_8"] / (df["target_roll_mean_96"] + 1e-6)
    if "target_roll_mean_16" in df.columns and "target_roll_mean_96" in df.columns:
        df["roll_ratio_16_96"] = df["target_roll_mean_16"] / (df["target_roll_mean_96"] + 1e-6)
    if "target_roll_mean_48" in df.columns and "target_roll_mean_336" in df.columns:
        df["roll_ratio_48_336"] = df["target_roll_mean_48"] / (df["target_roll_mean_336"] + 1e-6)
    if "target_ema_8" in df.columns and "target_ema_96" in df.columns:
        df["ema_ratio_8_96"] = df["target_ema_8"] / (df["target_ema_96"] + 1e-6)
    return df


# ── Custom split that returns a single extra step ─────────────────────────────

def split_with_extra_step(train_df: pd.DataFrame, feature_cols: list, step: int):
    step_col = f"target_step_{step}"
    df = train_df.copy()
    df[step_col] = df.groupby("route_id", sort=False)[TARGET_COL].shift(-step)

    supervised = df.dropna(subset=[step_col]).copy()
    print(f"  step_{step} supervised rows: {len(supervised)}")

    cols = feature_cols + ["timestamp", step_col]
    model_df = supervised[cols].rename(columns={"timestamp": "source_timestamp"})

    ts_max = model_df["source_timestamp"].max()
    ts_start = ts_max - pd.Timedelta(days=TRAIN_DAYS)
    model_df = model_df[model_df["source_timestamp"] >= ts_start].sort_values("source_timestamp")

    split_point = model_df["source_timestamp"].quantile(VALID_FRAC)
    fit_df  = model_df[model_df["source_timestamp"] <= split_point].copy()
    valid_df = model_df[model_df["source_timestamp"] > split_point].copy()

    if len(fit_df) > MAX_TRAIN_ROWS:
        fit_df = fit_df.sample(MAX_TRAIN_ROWS, random_state=RANDOM_STATE)

    print(f"  Fit rows: {len(fit_df)}  |  Valid rows: {len(valid_df)}")
    return fit_df[feature_cols].copy(), fit_df[step_col].copy(), \
           valid_df[feature_cols].copy(), valid_df[step_col].copy()


# ── Main ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print(f"STEP-11 + STEP-12 TRAINING: {EXPERIMENT_NAME}")
print("  Trains target_step_11 and target_step_12 for all 10 seeds")
print("  Then patches existing pkl files in-place")
print("  Then exports per-step-per-seed split pkls for GitHub upload")
print("=" * 60)
sys.stdout.flush()

models_dir = Path("models") / EXPERIMENT_NAME
pkl_files = sorted(models_dir.glob("*.pkl"))
if not pkl_files:
    raise FileNotFoundError(
        f"No pkl files found in {models_dir}. Run exp66_roll_ratio_struct_diversity.py first."
    )
print(f"\nFound {len(pkl_files)} existing pkl files:")
for p in pkl_files:
    print(f"  {p.name}")
sys.stdout.flush()

# ── Load data and build features (identical pipeline to exp66) ────────────────
print("\nLoading data + building features...")
sys.stdout.flush()
train_df_raw, _ = load_data()
train_df = make_features(train_df_raw, extended=True)
train_df = add_winning_features(train_df)
train_df = add_roll_ratio_features(train_df)
feature_cols = build_feature_cols(train_df)
print(f"  Feature count: {len(feature_cols)}")
sys.stdout.flush()

# Encode categoricals once (consistent categories across fit/valid)
cat_cols = [c for c in feature_cols if c.endswith("_id")]

def _encode_cats(X_fit, X_valid, cat_cols):
    X_fit, X_valid = X_fit.copy(), X_valid.copy()
    all_cats = {
        col: pd.concat([X_fit[col], X_valid[col]]).astype(str).unique()
        for col in cat_cols
    }
    for col, cats in all_cats.items():
        dtype = pd.CategoricalDtype(categories=cats)
        X_fit[col]   = X_fit[col].astype(str).astype(dtype)
        X_valid[col] = X_valid[col].astype(str).astype(dtype)
    return X_fit, X_valid


# ── Train steps 11 and 12 ─────────────────────────────────────────────────────
extra_step_models: dict[int, dict[str, lgb.LGBMRegressor]] = {}
# extra_step_models[step][pkl_name] = trained model

for extra_step in (11, 12):
    step_col = f"target_step_{extra_step}"
    print(f"\n{'-' * 50}")
    print(f"Training {step_col} ...")
    sys.stdout.flush()

    # Check if already present in all pkls
    already_done = [p for p in pkl_files if step_col in joblib.load(p)]
    if len(already_done) == len(pkl_files):
        print(f"  All pkl files already contain {step_col}. Re-training anyway.")

    X_fit_raw, y_fit, X_valid_raw, y_valid = split_with_extra_step(
        train_df, feature_cols, extra_step
    )
    X_fit_enc, X_valid_enc = _encode_cats(X_fit_raw, X_valid_raw, cat_cols)

    step_models: dict[str, lgb.LGBMRegressor] = {}

    for label, seeds, cs, ss in GROUPS:
        print(f"\n  >>> {label}: cs={cs} ss={ss} <<<")
        sys.stdout.flush()
        for seed in seeds:
            pkl_name = f"lgb_{label}_seed{seed}.pkl"
            t0 = datetime.now()
            print(f"    seed={seed} [{t0.strftime('%H:%M:%S')}]...", end=" ")
            sys.stdout.flush()

            params = {
                **BASE_LGB,
                "objective": "quantile",
                "alpha": ALPHA,
                "random_state": seed,
                "colsample_bytree": cs,
                "subsample": ss,
            }
            m = lgb.LGBMRegressor(**params)
            m.fit(
                X_fit_enc, y_fit,
                eval_set=[(X_valid_enc, y_valid)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            val_pred = np.clip(m.predict(X_valid_enc), 0, None)
            wape = np.abs(val_pred - y_valid).sum() / (y_valid.sum() + 1e-8)
            elapsed = (datetime.now() - t0).seconds
            print(f"iter={m.best_iteration_:4d}  WAPE={wape:.4f}  [{elapsed}s]")
            sys.stdout.flush()
            step_models[pkl_name] = m

    extra_step_models[extra_step] = step_models

    # ── Patch existing ensemble pkl files ─────────────────────────────────────
    print(f"\n  [Patching pkl files with {step_col}]")
    sys.stdout.flush()
    for pkl_path in pkl_files:
        ensemble_dict = joblib.load(pkl_path)
        m = step_models.get(pkl_path.name)
        if m is None:
            print(f"    SKIP {pkl_path.name} — no model trained (unexpected)")
            continue
        ensemble_dict[step_col] = m
        joblib.dump(ensemble_dict, pkl_path)
        steps_present = sorted(k for k in ensemble_dict if k.startswith("target_step_"))
        print(f"    Patched {pkl_path.name}: {steps_present}")
    sys.stdout.flush()

    # ── Ensemble validation ───────────────────────────────────────────────────
    preds_all = []
    for pkl_path in sorted(pkl_files):
        ens = joblib.load(pkl_path)
        preds_all.append(np.clip(ens[step_col].predict(X_valid_enc), 0, None))
    ens_pred = np.mean(preds_all, axis=0)
    wape_ens = np.abs(ens_pred - y_valid).sum() / (y_valid.sum() + 1e-8)
    bias_ens = ens_pred.sum() / (y_valid.sum() + 1e-8) - 1.0
    print(f"\n  Ensemble {step_col}: WAPE={wape_ens:.4f}  RBias={bias_ens:+.4f}")
    sys.stdout.flush()


# ── Export per-step-per-seed split pkl files for GitHub ──────────────────────
print("\n" + "=" * 60)
print("Exporting per-step-per-seed split pkl files to ml_exp66_prediction/models/ ...")
sys.stdout.flush()

split_dir = Path("ml_exp66_prediction") / "models"
split_dir.mkdir(parents=True, exist_ok=True)

for pkl_path in sorted(pkl_files):
    ensemble_dict = joblib.load(pkl_path)
    seed_stem = pkl_path.stem  # e.g. lgb_A_cs08_seed42
    for step_key, model in ensemble_dict.items():
        if not step_key.startswith("target_step_"):
            continue
        step_num = int(step_key.replace("target_step_", ""))
        out_name = f"{seed_stem}_step{step_num:02d}.pkl"
        out_path = split_dir / out_name
        joblib.dump(model, out_path)
    steps_exported = sorted(
        int(k.replace("target_step_", ""))
        for k in ensemble_dict if k.startswith("target_step_")
    )
    print(f"  {pkl_path.name} -> steps {steps_exported}")
    sys.stdout.flush()

all_split_pkls = sorted(split_dir.glob("*.pkl"))
print(f"\nTotal split pkl files: {len(all_split_pkls)}")
print(f"Location: {split_dir.resolve()}")
sys.stdout.flush()

# ── Log ──────────────────────────────────────────────────────────────────────
log_path = Path("experiments.json")
history = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
history.append({
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "name": f"{EXPERIMENT_NAME}_step11_step12",
    "params": dict(
        groups=[(g[0], g[1], g[2], g[3]) for g in GROUPS],
        alpha=ALPHA,
        extra_steps=[11, 12],
        split_pkls_dir=str(split_dir),
    ),
    "note": "Trained step_11 and step_12, patched ensemble pkls, exported per-step split pkls.",
})
log_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
print("  Logged to experiments.json")


# ── Compute non-conformity scores (normalized) for all 12 steps ──────────────
# Structure mirrors data/non_conformity_scores_norm_allsteps.csv:
#   columns: route_id, horizon, step, score
# Uses the same validation split as training (21 days / 80%).
# score = |y_true - y_hat| / (y_hat + 1e-6)
print("\n" + "=" * 60)
print("Computing non-conformity scores (all 12 steps)...")
sys.stdout.flush()

HORIZON_LABELS = {
    1:  "-1.5-0.5h",
    2:  "-1-1h",
    3:  "-0.5-1.5h",
    4:  "0-2h",
    5:  "0.5-2.5h",
    6:  "1-3h",
    7:  "1.5-3.5h",
    8:  "2-4h",
    9:  "2.5-4.5h",
    10: "3-5h",
    11: "3.5-5.5h",
    12: "4-6h",
}

NC_OUT_PATH = Path("ml_exp66_prediction") / "non_conformity_scores_norm_allsteps.csv"

# Build validation features in the same way as training
# train_df already has all features built above; we just need the validation rows.
# Rebuild the time-based split to get X_valid with full feature set.
_model_df = train_df[feature_cols + ["timestamp"]].rename(
    columns={"timestamp": "source_timestamp"}
).copy()
_ts_max = _model_df["source_timestamp"].max()
_ts_start = _ts_max - pd.Timedelta(days=TRAIN_DAYS)
_model_df = _model_df[_model_df["source_timestamp"] >= _ts_start].sort_values("source_timestamp")
_split_point = _model_df["source_timestamp"].quantile(VALID_FRAC)
_valid_idx = _model_df[_model_df["source_timestamp"] > _split_point].index

valid_full = train_df.loc[_valid_idx].copy()
X_valid_nc = valid_full[feature_cols].copy()
route_ids_nc = valid_full["route_id"].values

# Build future-target columns for all 12 steps (in case train_df doesn't have them)
_g = train_df.groupby("route_id", sort=False)
for _k in range(1, 13):
    _col = f"target_step_{_k}"
    if _col not in valid_full.columns:
        valid_full[_col] = _g[TARGET_COL].shift(-_k).loc[_valid_idx]

# Encode categoricals on the validation set
for col in cat_cols:
    _cats = X_valid_nc[col].astype(str).unique()
    X_valid_nc[col] = X_valid_nc[col].astype(str).astype(pd.CategoricalDtype(categories=_cats))

# Load freshly-patched ensemble pkls (now contain all 12 steps)
ensembles_nc = [joblib.load(p) for p in sorted(pkl_files)]
available_steps_nc = [
    k for k in range(1, 13)
    if all(f"target_step_{k}" in m for m in ensembles_nc)
]
print(f"  Validation rows: {len(valid_full)}")
print(f"  Scoring steps: {available_steps_nc}")
sys.stdout.flush()

nc_chunks = []
for k in available_steps_nc:
    step_key = f"target_step_{k}"
    label = HORIZON_LABELS[k]

    target_vals = valid_full[step_key].values
    mask = ~np.isnan(target_vals)
    y_true = target_vals[mask]
    X_step = X_valid_nc.iloc[mask]
    rids_step = route_ids_nc[mask]

    preds = np.mean(
        [np.clip(m[step_key].predict(X_step), 0, None) for m in ensembles_nc],
        axis=0,
    )
    scores = np.abs(y_true - preds) / (preds + 1e-6)

    wape = np.abs(y_true - preds).sum() / (y_true.sum() + 1e-8)
    print(
        f"  step_{k:2d} [{label:>12s}] n={mask.sum():>7d}  "
        f"WAPE={wape:.4f}  median={np.median(scores):.4f}  p95={np.percentile(scores, 95):.4f}"
    )
    sys.stdout.flush()

    nc_chunks.append(pd.DataFrame({
        "route_id": rids_step.astype(int),
        "horizon": label,
        "step": k,
        "score": np.round(scores, 6),
    }))

NC_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
nc_df = pd.concat(nc_chunks, ignore_index=True)
nc_df.to_csv(NC_OUT_PATH, index=False)
print(f"\nSaved {len(nc_df)} rows to {NC_OUT_PATH}")
sys.stdout.flush()


print("\n" + "=" * 60)
print(f"DONE:  exp66 step_11 + step_12 complete.")
print(f"  All 10 ensemble pkls now contain steps 1-12.")
print(f"  {len(all_split_pkls)} individual step pkls in {split_dir}/")
print(f"  Non-conformity scores: {NC_OUT_PATH}")
print("=" * 60)
