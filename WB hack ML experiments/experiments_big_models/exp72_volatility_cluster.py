"""Experiment 72: 3-cluster by route volatility (CV = std/mean).

Routes have very different demand stability. A "stable" route with CV=0.2
follows regular weekly cycles; a "volatile" route with CV=2.0 is unpredictable.
These two types of routes need different feature weighting: stable routes
benefit from long seasonal patterns, volatile routes from recent short windows.

Cluster by coefficient of variation per route:
  Stable:   CV < p33 (most predictable routes)
  Moderate: CV in [p33, p67)
  Volatile: CV >= p67 (hardest to predict)

Routes with mean_target ≈ 0 get CV = inf => always go to "volatile" bucket.
This is desired because zero-heavy sparse routes are genuinely hard to predict.

Each cluster: 3 seeds, cs=0.5, ss=0.7, leaves=511, alpha=0.55, 21d, 158 feats.
"""

import json, sys
from datetime import datetime
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from config import FUTURE_TARGET_COLS, TARGET_COL
from data import build_feature_cols, create_future_targets, encode_categoricals, load_data, split_data
from features import make_features
from metrics import WapePlusRbias
from train import build_submission, predict_steps

EXPERIMENT_NAME = "exp72_volatility_cluster"
SEEDS = [42, 123, 456]
ALPHA = 0.55
COLSAMPLE = 0.5
SUBSAMPLE = 0.7

BASE_LGB = dict(n_estimators=5000, learning_rate=0.05, num_leaves=511,
    min_child_samples=10, subsample_freq=1, reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1, verbose=-1)

print("=" * 60)
print(f"EXPERIMENT: {EXPERIMENT_NAME}")
print(f"  3 volatility clusters (stable/moderate/volatile by CV) x {len(SEEDS)} seeds each")
print(f"  cs={COLSAMPLE} ss={SUBSAMPLE} leaves=511 alpha={ALPHA} 21d | 158 feats")
print("=" * 60)
sys.stdout.flush()


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


def add_roll_ratio_features(df):
    if "target_roll_mean_8" in df.columns and "target_roll_mean_96" in df.columns:
        df["roll_ratio_8_96"] = df["target_roll_mean_8"] / (df["target_roll_mean_96"] + 1e-6)
    if "target_roll_mean_16" in df.columns and "target_roll_mean_96" in df.columns:
        df["roll_ratio_16_96"] = df["target_roll_mean_16"] / (df["target_roll_mean_96"] + 1e-6)
    if "target_roll_mean_48" in df.columns and "target_roll_mean_336" in df.columns:
        df["roll_ratio_48_336"] = df["target_roll_mean_48"] / (df["target_roll_mean_336"] + 1e-6)
    if "target_ema_8" in df.columns and "target_ema_96" in df.columns:
        df["ema_ratio_8_96"] = df["target_ema_8"] / (df["target_ema_96"] + 1e-6)
    return df


def train_cluster_models(X_fit, y_fit, X_valid, y_valid, seeds, base_params):
    vp_list, model_list = [], []
    for seed in seeds:
        params = {**base_params, "random_state": seed}
        models = {}
        for step_col in FUTURE_TARGET_COLS:
            m = lgb.LGBMRegressor(**params)
            m.fit(X_fit, y_fit[step_col], eval_set=[(X_valid, y_valid[step_col])],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
            models[step_col] = m
        vp = predict_steps(models, X_valid)
        vp_list.append(vp)
        model_list.append(models)
        w, r, t = WapePlusRbias().calculate_components(y_valid, vp)
        print(f"    seed={seed}: WAPE={w:.4f} RBias={r:.4f} Total={t:.4f}")
        sys.stdout.flush()
    return vp_list, model_list


print("Loading + building features...")
sys.stdout.flush()
train_df_raw, test_df = load_data()
train_df = make_features(train_df_raw, extended=True)
train_df = add_winning_features(train_df)
train_df = add_roll_ratio_features(train_df)
train_df = create_future_targets(train_df)
feature_cols = build_feature_cols(train_df)
print(f"Feature count: {len(feature_cols)}")
sys.stdout.flush()

# ── Compute volatility clusters ────────────────────────────────────────────────
route_stats = train_df_raw.groupby("route_id")[TARGET_COL].agg(["mean", "std"]).rename(
    columns={"mean": "mean_t", "std": "std_t"})
route_stats["cv"] = route_stats["std_t"] / (route_stats["mean_t"] + 1e-6)

cv_finite = route_stats["cv"][route_stats["mean_t"] > 0]
q33 = cv_finite.quantile(1/3)
q67 = cv_finite.quantile(2/3)
print(f"CV thresholds: p33={q33:.3f}  p67={q67:.3f}")

def assign_vol_cluster(route_id):
    cv = route_stats.loc[route_id, "cv"] if route_id in route_stats.index else q67
    if cv < q33:   return "stable"
    elif cv < q67: return "moderate"
    else:          return "volatile"

route_cluster = {r: assign_vol_cluster(r) for r in route_stats.index}
clusters = ["stable", "moderate", "volatile"]
for c in clusters:
    n = sum(1 for v in route_cluster.values() if v == c)
    cvs = route_stats["cv"][[r for r, cl in route_cluster.items() if cl == c]]
    print(f"  Cluster '{c}': {n} routes  CV=[{cvs.min():.3f}, {cvs.max():.3f}]")
sys.stdout.flush()

# ── Global split ───────────────────────────────────────────────────────────────
X_fit_g, y_fit_g, X_valid_g, y_valid_g, X_test, inference_ts, _ = split_data(
    train_df, feature_cols, train_days=21)

# Map cluster BEFORE encode_categoricals (route_id still has original values)
X_fit_g["_cluster"]   = X_fit_g["route_id"].map(route_cluster)
X_valid_g["_cluster"] = X_valid_g["route_id"].map(route_cluster)
X_test["_cluster"]    = X_test["route_id"].map(route_cluster)

X_fit_g, X_valid_g, X_test, _ = encode_categoricals(X_fit_g, X_valid_g, X_test, feature_cols)
print(f"Fit={len(X_fit_g):,}  Valid={len(X_valid_g):,}  Test={len(X_test):,}")
sys.stdout.flush()

metric = WapePlusRbias()
models_dir = Path("models") / EXPERIMENT_NAME
models_dir.mkdir(parents=True, exist_ok=True)
base_params = {**BASE_LGB, "objective": "quantile", "alpha": ALPHA,
               "colsample_bytree": COLSAMPLE, "subsample": SUBSAMPLE}

valid_pred_parts, test_pred_parts = [], []

for cluster in clusters:
    mask_fit   = X_fit_g["_cluster"] == cluster
    mask_valid = X_valid_g["_cluster"] == cluster
    mask_test  = X_test["_cluster"] == cluster

    Xf = X_fit_g[mask_fit].drop(columns=["_cluster"])
    yf = y_fit_g.loc[Xf.index]
    Xv = X_valid_g[mask_valid].drop(columns=["_cluster"])
    yv = y_valid_g.loc[Xv.index]
    Xt = X_test[mask_test].drop(columns=["_cluster"])

    print(f"\n>>> Cluster '{cluster}': fit={len(Xf):,}  valid={len(Xv):,}  test={len(Xt):,} <<<")
    sys.stdout.flush()

    t0 = datetime.now()
    vp_list, model_list = train_cluster_models(Xf, yf, Xv, yv, SEEDS, base_params)
    print(f"  Training time: {(datetime.now()-t0).seconds}s")
    sys.stdout.flush()

    vp_ens = pd.DataFrame(np.mean([p.values for p in vp_list], axis=0),
                          columns=FUTURE_TARGET_COLS, index=Xv.index).clip(lower=0)
    tp_list = [predict_steps(m, Xt) for m in model_list]
    tp_ens  = pd.DataFrame(np.mean([p.values for p in tp_list], axis=0),
                           columns=FUTURE_TARGET_COLS, index=Xt.index).clip(lower=0)

    w, r, t = metric.calculate_components(yv, vp_ens)
    print(f"  [{cluster}] ensemble: WAPE={w:.4f} RBias={r:.4f} Total={t:.4f}")
    sys.stdout.flush()

    valid_pred_parts.append((vp_ens, yv))
    test_pred_parts.append(tp_ens)
    for i, m in enumerate(model_list):
        joblib.dump(m, models_dir / f"lgb_{cluster}_seed{SEEDS[i]}.pkl")

valid_pred_all = pd.concat([v for v, _ in valid_pred_parts]).sort_index()
y_valid_all    = pd.concat([y for _, y in valid_pred_parts]).sort_index()
test_pred_all  = pd.concat(test_pred_parts).sort_index()

w_raw, r_raw, t_raw = metric.calculate_components(y_valid_all, valid_pred_all)
print(f"\n[Merged Raw]  WAPE={w_raw:.4f} RBias={r_raw:.4f} Total={t_raw:.4f}")

factors = {}
valid_cal, test_cal = valid_pred_all.copy(), test_pred_all.copy()
for col in FUTURE_TARGET_COLS:
    f = float(y_valid_all[col].sum()) / max(float(valid_pred_all[col].sum()), 1e-9)
    valid_cal[col] = (valid_pred_all[col] * f).clip(lower=0)
    test_cal[col]  = (test_pred_all[col]  * f).clip(lower=0)
    factors[col] = round(f, 4)
w_cal, r_cal, t_cal = metric.calculate_components(y_valid_all, valid_cal)
print(f"[Calibrated]  WAPE={w_cal:.4f} RBias={r_cal:.4f} Total={t_cal:.4f}")

sub = build_submission(test_cal, X_test.drop(columns=["_cluster"]), inference_ts, test_df)
sub.to_csv(f"submission_team_{EXPERIMENT_NAME}.csv", index=False)
print(f"Saved: submission_team_{EXPERIMENT_NAME}.csv  ({len(sub)} rows)")

exp_path = Path("experiments.json")
exps = json.loads(exp_path.read_text(encoding="utf-8"))
exps.append({"timestamp": datetime.now().isoformat(timespec="seconds"), "name": EXPERIMENT_NAME,
    "params": {"clusters": 3, "cluster_type": "cv_quantile",
               "cv_thresholds": {"p33": round(q33,4), "p67": round(q67,4)},
               "seeds": SEEDS, "alpha": ALPHA, "num_leaves": 511,
               "colsample_bytree": COLSAMPLE, "subsample": SUBSAMPLE,
               "n_models_total": 3 * len(SEEDS), "train_days": 21,
               "features": "158", "calibration": "global_per_step"},
    "wape": round(w_cal,6), "rbias": round(r_cal,6), "total": round(t_cal,6),
    "note": (f"3 volatility clusters by CV=std/mean (stable/moderate/volatile). "
             f"p33_cv={q33:.4f} p67_cv={q67:.4f}. Raw:{t_raw:.4f} Cal:{t_cal:.4f}.")})
exp_path.write_text(json.dumps(exps, indent=2, ensure_ascii=False), encoding="utf-8")

print("\n" + "=" * 60)
print(f"EXPERIMENT {EXPERIMENT_NAME} COMPLETE")
print(f"  Raw:        WAPE={w_raw:.4f}  RBias={r_raw:.4f}  Total={t_raw:.4f}")
print(f"  Calibrated: WAPE={w_cal:.4f}  RBias={r_cal:.4f}  Total={t_cal:.4f}")
print("=" * 60)
