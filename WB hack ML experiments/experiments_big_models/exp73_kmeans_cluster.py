"""Experiment 73: 3-cluster KMeans on route profile [mean, CV, zero_frac].

Data-driven clustering instead of hand-crafted thresholds.
Routes are embedded in 3D space: (mean_target, CV=std/mean, zero_frac).
MinMaxScaler normalizes each dimension before KMeans.

This lets the algorithm find natural groupings that may not align with
clean volume or CV quantile cuts — e.g., a high-volume but very sparse
route (returns hub? delivery exception accumulation?) lands in a different
cluster than a high-volume always-active route.

k=3 clusters: separate model per cluster, 3 seeds each, 158 feats.
Same LGB config as exp61 best (cs=0.5, ss=0.7, leaves=511, alpha=0.55).
"""

import json, sys
from datetime import datetime
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, ".")
from config import FUTURE_TARGET_COLS, TARGET_COL
from data import build_feature_cols, create_future_targets, encode_categoricals, load_data, split_data
from features import make_features
from metrics import WapePlusRbias
from train import build_submission, predict_steps

EXPERIMENT_NAME = "exp73_kmeans_cluster"
N_CLUSTERS = 3
SEEDS = [42, 123, 456]
ALPHA = 0.55
COLSAMPLE = 0.5
SUBSAMPLE = 0.7

BASE_LGB = dict(n_estimators=5000, learning_rate=0.05, num_leaves=511,
    min_child_samples=10, subsample_freq=1, reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1, verbose=-1)

print("=" * 60)
print(f"EXPERIMENT: {EXPERIMENT_NAME}")
print(f"  KMeans(k=3) on [mean, CV, zero_frac] profile x {len(SEEDS)} seeds each")
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

# ── Compute KMeans clusters on route profile ───────────────────────────────────
route_stats = train_df_raw.groupby("route_id")[TARGET_COL].agg(
    mean_t="mean", std_t="std").fillna(0)
route_stats["cv"]        = route_stats["std_t"] / (route_stats["mean_t"] + 1e-6)
route_stats["zero_frac"] = (train_df_raw.groupby("route_id")[TARGET_COL]
                             .apply(lambda x: (x == 0).mean()))

profile = route_stats[["mean_t", "cv", "zero_frac"]].values
scaler  = MinMaxScaler()
profile_scaled = scaler.fit_transform(profile)

km = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init=20)
km.fit(profile_scaled)
labels = km.labels_  # 0, 1, 2

route_ids = route_stats.index.tolist()
route_cluster = {r: f"cluster_{labels[i]}" for i, r in enumerate(route_ids)}
clusters = [f"cluster_{i}" for i in range(N_CLUSTERS)]

for c in clusters:
    idx = [r for r, cl in route_cluster.items() if cl == c]
    sub_stats = route_stats.loc[idx]
    print(f"  {c}: {len(idx)} routes | "
          f"mean=[{sub_stats['mean_t'].min():.1f},{sub_stats['mean_t'].max():.1f}] "
          f"cv=[{sub_stats['cv'].min():.2f},{sub_stats['cv'].max():.2f}] "
          f"zero_frac=[{sub_stats['zero_frac'].min():.3f},{sub_stats['zero_frac'].max():.3f}]")
sys.stdout.flush()

# ── Global split ───────────────────────────────────────────────────────────────
X_fit_g, y_fit_g, X_valid_g, y_valid_g, X_test, inference_ts, _ = split_data(
    train_df, feature_cols, train_days=21)

# Map cluster BEFORE encode_categoricals (route_id still has original values)
X_fit_g["_cluster"]   = X_fit_g["route_id"].map(route_cluster).fillna("cluster_0")
X_valid_g["_cluster"] = X_valid_g["route_id"].map(route_cluster).fillna("cluster_0")
X_test["_cluster"]    = X_test["route_id"].map(route_cluster).fillna("cluster_0")

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

    print(f"\n>>> {cluster}: fit={len(Xf):,}  valid={len(Xv):,}  test={len(Xt):,} <<<")
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

centers = scaler.inverse_transform(km.cluster_centers_)
cluster_info = {f"cluster_{i}": {"mean_t": round(float(c[0]),2),
    "cv": round(float(c[1]),3), "zero_frac": round(float(c[2]),3)}
    for i, c in enumerate(centers)}

sub = build_submission(test_cal, X_test.drop(columns=["_cluster"]), inference_ts, test_df)
sub.to_csv(f"submission_team_{EXPERIMENT_NAME}.csv", index=False)
print(f"Saved: submission_team_{EXPERIMENT_NAME}.csv  ({len(sub)} rows)")

exp_path = Path("experiments.json")
exps = json.loads(exp_path.read_text(encoding="utf-8"))
exps.append({"timestamp": datetime.now().isoformat(timespec="seconds"), "name": EXPERIMENT_NAME,
    "params": {"clusters": N_CLUSTERS, "cluster_type": "kmeans_profile",
               "cluster_features": ["mean_t", "cv", "zero_frac"],
               "cluster_centers": cluster_info, "seeds": SEEDS,
               "alpha": ALPHA, "num_leaves": 511, "colsample_bytree": COLSAMPLE,
               "subsample": SUBSAMPLE, "n_models_total": N_CLUSTERS * len(SEEDS),
               "train_days": 21, "features": "158", "calibration": "global_per_step"},
    "wape": round(w_cal,6), "rbias": round(r_cal,6), "total": round(t_cal,6),
    "note": (f"KMeans(k=3) on route profile [mean, CV, zero_frac] (MinMaxScaled). "
             f"Raw:{t_raw:.4f} Cal:{t_cal:.4f}.")})
exp_path.write_text(json.dumps(exps, indent=2, ensure_ascii=False), encoding="utf-8")

print("\n" + "=" * 60)
print(f"EXPERIMENT {EXPERIMENT_NAME} COMPLETE")
print(f"  Raw:        WAPE={w_raw:.4f}  RBias={r_raw:.4f}  Total={t_raw:.4f}")
print(f"  Calibrated: WAPE={w_cal:.4f}  RBias={r_cal:.4f}  Total={t_cal:.4f}")
print("=" * 60)
