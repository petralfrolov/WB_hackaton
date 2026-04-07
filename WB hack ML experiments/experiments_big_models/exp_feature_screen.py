"""Fast feature screening: 1 seed, 1 step (target_step_1).

All features precomputed once. Each test = different column subset.
~20-40 seconds per test.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import json
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, ".")
from config import TARGET_COL, TRAIN_DAYS, MAX_TRAIN_ROWS, VALID_FRAC, RANDOM_STATE
from data import load_data, create_future_targets, build_feature_cols
from features import make_features

STEP_COL = "target_step_1"

FAST_LGB = dict(
    n_estimators=1500, learning_rate=0.05, num_leaves=63,
    min_child_samples=30, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    objective="quantile", alpha=0.55,
    n_jobs=-1, verbose=-1, random_state=42,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE GROUP BUILDERS — each returns (df_with_new_cols, list_of_new_col_names)
# ═══════════════════════════════════════════════════════════════════════════════

def feat_rolling_cv(df):
    new_cols = []
    for w in [4, 8, 48]:
        mean_col, std_col = f"target_roll_mean_{w}", f"target_roll_std_{w}"
        if mean_col in df.columns and std_col in df.columns:
            name = f"target_cv_{w}"
            df[name] = df[std_col] / (df[mean_col] + 1e-6)
            new_cols.append(name)
    return df, new_cols

def feat_fourier_harmonics(df):
    new_cols = []
    for k in [2, 3]:
        for base, period in [("hour", 24), ("day_of_week", 7)]:
            for fn, label in [(np.sin, "sin"), (np.cos, "cos")]:
                name = f"{base}_{label}_{k}"
                df[name] = fn(2 * np.pi * k * df[base] / period)
                new_cols.append(name)
    for k in [1, 2, 3]:
        for fn, label in [(np.sin, "sin"), (np.cos, "cos")]:
            name = f"slot_{label}_{k}"
            df[name] = fn(2 * np.pi * k * df["halfhour_slot"] / 48)
            new_cols.append(name)
    return df, new_cols

def feat_dest_warehouse_agg(df):
    new_cols = []
    if "office_to_id" in df.columns:
        agg = (df.groupby(["office_to_id", "timestamp"])[TARGET_COL]
               .agg(dest_target_mean="mean", dest_target_sum="sum")
               .reset_index())
        df = df.merge(agg, on=["office_to_id", "timestamp"], how="left")
        new_cols = ["dest_target_mean", "dest_target_sum"]
    return df, new_cols

def feat_lag_ratios(df):
    new_cols = []
    pairs = [("target_lag_1", "target_lag_48", "lag_ratio_1_48", "/"),
             ("target_lag_1", "target_lag_336", "lag_ratio_1_336", "/"),
             ("target_lag_1", "target_ema_8", "momentum_1_ema8", "-"),
             ("target_lag_1", "target_ema_24", "momentum_1_ema24", "-")]
    for a, b, name, op in pairs:
        if a in df.columns and b in df.columns:
            if op == "/":
                df[name] = df[a] / (df[b] + 1e-6)
            else:
                df[name] = df[a] - df[b]
            new_cols.append(name)
    return df, new_cols

def feat_time_interactions(df):
    df["time_of_day"] = df["hour"] + df["timestamp"].dt.minute / 60.0
    df["dow_x_time"] = df["day_of_week"] * 48 + df["halfhour_slot"]
    return df, ["time_of_day", "dow_x_time"]

def feat_route_office_rank(df):
    new_cols = []
    if "route_target_mean" in df.columns:
        df["route_office_rank"] = df.groupby("office_from_id")["route_target_mean"].rank(pct=True)
        new_cols.append("route_office_rank")
    return df, new_cols

def feat_zero_share(df):
    new_cols = []
    g = df.groupby("route_id", sort=False)
    is_zero = (g[TARGET_COL].shift(1) == 0).astype(np.float32)
    for w in [48, 336]:
        name = f"zero_share_{w}"
        df[name] = is_zero.groupby(df["route_id"], sort=False).transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )
        new_cols.append(name)
    return df, new_cols

def feat_recent_max_ratio(df):
    new_cols = []
    if "target_lag_1" in df.columns:
        for w in [8, 48]:
            max_col = f"target_roll_max_{w}"
            if max_col in df.columns:
                name = f"lag1_to_max_{w}"
                df[name] = df["target_lag_1"] / (df[max_col] + 1e-6)
                new_cols.append(name)
    return df, new_cols

def feat_target_range_ratio(df):
    new_cols = []
    for w in [8, 48]:
        max_c, min_c, mean_c = f"target_roll_max_{w}", f"target_roll_min_{w}", f"target_roll_mean_{w}"
        if all(c in df.columns for c in [max_c, min_c, mean_c]):
            name = f"target_range_ratio_{w}"
            df[name] = (df[max_c] - df[min_c]) / (df[mean_c] + 1e-6)
            new_cols.append(name)
    return df, new_cols

def feat_weekly_diff(df):
    new_cols = []
    if "target_lag_1" in df.columns and "target_lag_336" in df.columns:
        df["weekly_diff"] = df["target_lag_1"] - df["target_lag_336"]
        new_cols.append("weekly_diff")
    if "target_lag_1" in df.columns and "target_lag_672" in df.columns:
        df["biweekly_diff"] = df["target_lag_1"] - df["target_lag_672"]
        new_cols.append("biweekly_diff")
    return df, new_cols

def feat_office_route_ratio(df):
    new_cols = []
    if "target_lag_1" in df.columns and "office_target_sum" in df.columns:
        df["office_lag1_sum"] = df.groupby("office_from_id")["office_target_sum"].shift(1)
        df["route_share_office"] = df["target_lag_1"] / (df["office_lag1_sum"] + 1e-6)
        new_cols.extend(["office_lag1_sum", "route_share_office"])
    return df, new_cols

def feat_peak_indicator(df):
    h = df["hour"]
    df["is_morning_peak"] = ((h >= 8) & (h <= 11)).astype(np.int8)
    df["is_evening_peak"] = ((h >= 16) & (h <= 19)).astype(np.int8)
    df["is_night"] = ((h >= 22) | (h <= 5)).astype(np.int8)
    df["is_weekday_daytime"] = ((df["is_weekend"] == 0) & (h >= 8) & (h <= 18)).astype(np.int8)
    return df, ["is_morning_peak", "is_evening_peak", "is_night", "is_weekday_daytime"]


# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_GROUPS = {
    "rolling_cv":         feat_rolling_cv,
    "fourier":            feat_fourier_harmonics,
    "dest_warehouse":     feat_dest_warehouse_agg,
    "lag_ratios":         feat_lag_ratios,
    "time_interact":      feat_time_interactions,
    "route_rank":         feat_route_office_rank,
    "zero_share":         feat_zero_share,
    "recent_max_ratio":   feat_recent_max_ratio,
    "range_ratio":        feat_target_range_ratio,
    "weekly_diff":        feat_weekly_diff,
    "office_route_ratio": feat_office_route_ratio,
    "peak_indicator":     feat_peak_indicator,
}


def encode_cats(X_fit, X_valid, feat_cols):
    cat_cols = [c for c in feat_cols if c.endswith("_id")]
    for col in cat_cols:
        cats = pd.concat([X_fit[col], X_valid[col]]).astype(str).unique()
        dtype = pd.CategoricalDtype(categories=cats)
        X_fit[col] = X_fit[col].astype(str).astype(dtype)
        X_valid[col] = X_valid[col].astype(str).astype(dtype)
    return X_fit, X_valid, cat_cols


def train_eval(X_fit, y_fit, X_valid, y_valid):
    m = lgb.LGBMRegressor(**FAST_LGB)
    m.fit(
        X_fit, y_fit,
        eval_set=[(X_valid, y_valid)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    vs = m.best_score_.get("valid_0", {})
    lgb_score = next(iter(vs.values()), float("nan"))

    pred = m.predict(X_valid).clip(0)
    factor = y_valid.sum() / (pred.sum() + 1e-12)
    pred *= factor

    true_arr = y_valid.values
    wape = np.abs(pred - true_arr).sum() / true_arr.sum()
    return float(wape), m.best_iteration_, lgb_score


def main():
    print("=" * 70)
    print("FEATURE SCREENING — 1 seed, 1 step (target_step_1)")
    print("  Precompute all features, test by column selection")
    print("=" * 70)
    sys.stdout.flush()

    # ── 1. Load + build all features at once ──────────────────────────────────
    print("\nLoading data...")
    sys.stdout.flush()
    train_df, _ = load_data()

    print("Building base features (extended + deconv)...")
    sys.stdout.flush()
    train_df = make_features(train_df, extended=True)
    train_df = create_future_targets(train_df)

    base_feat_cols = build_feature_cols(train_df)
    print(f"Base features: {len(base_feat_cols)}")

    print("Adding all extra feature groups...")
    sys.stdout.flush()
    group_cols = {}
    for name, fn in FEATURE_GROUPS.items():
        t0 = time.time()
        train_df, new_cols = fn(train_df)
        group_cols[name] = new_cols
        print(f"  {name}: +{len(new_cols)} cols  ({time.time()-t0:.1f}s)")
        sys.stdout.flush()

    # ── 2. Single split (once!) ───────────────────────────────────────────────
    print("\nSplitting data...")
    sys.stdout.flush()
    supervised = train_df.dropna(subset=[STEP_COL]).copy()
    all_feat_cols = base_feat_cols + [c for cols in group_cols.values() for c in cols]
    keep_cols = list(set(all_feat_cols + ["timestamp", STEP_COL]))
    model_df = supervised[keep_cols].copy()
    del supervised, train_df

    model_df = model_df.rename(columns={"timestamp": "ts"})
    ts_max = model_df["ts"].max()
    ts_start = ts_max - pd.Timedelta(days=TRAIN_DAYS)
    model_df = model_df[model_df["ts"] >= ts_start].sort_values("ts")

    split_point = model_df["ts"].quantile(VALID_FRAC)
    fit_mask = model_df["ts"] <= split_point
    valid_mask = ~fit_mask

    fit_df = model_df[fit_mask]
    valid_df = model_df[valid_mask]
    if len(fit_df) > MAX_TRAIN_ROWS:
        fit_df = fit_df.sample(MAX_TRAIN_ROWS, random_state=RANDOM_STATE)

    y_fit = fit_df[STEP_COL]
    y_valid = valid_df[STEP_COL]
    print(f"Fit: {len(fit_df)}  Valid: {len(valid_df)}")
    sys.stdout.flush()

    # ── 3. Baseline ───────────────────────────────────────────────────────────
    print(f"\n{'='*50}\n  BASELINE ({len(base_feat_cols)} features)\n{'='*50}")
    sys.stdout.flush()

    Xf, Xv, cat_cols = encode_cats(
        fit_df[base_feat_cols].copy(), valid_df[base_feat_cols].copy(), base_feat_cols)
    t0 = time.time()
    bl_wape, bl_iter, bl_lgb = train_eval(Xf, y_fit, Xv, y_valid)
    bl_time = time.time() - t0
    print(f"  WAPE={bl_wape:.6f}  best_iter={bl_iter}  lgb={bl_lgb:.4f}  ({bl_time:.0f}s)")
    sys.stdout.flush()

    results = [{"test": "baseline", "n_features": len(base_feat_cols),
                "wape": round(bl_wape, 6), "delta": 0.0,
                "best_iter": bl_iter, "time_s": round(bl_time, 1)}]

    # ── 4. Test each group = baseline + group_cols ────────────────────────────
    for name, new_cols in group_cols.items():
        if not new_cols:
            continue
        cols = base_feat_cols + new_cols
        print(f"\n--- {name} (+{len(new_cols)} = {len(cols)} feats) ---")
        sys.stdout.flush()

        Xf, Xv, _ = encode_cats(fit_df[cols].copy(), valid_df[cols].copy(), cols)
        t0 = time.time()
        wape, best_iter, lgb_score = train_eval(Xf, y_fit, Xv, y_valid)
        elapsed = time.time() - t0
        delta = wape - bl_wape

        marker = " *** BETTER" if delta < -1e-6 else ""
        print(f"  WAPE={wape:.6f}  Δ={delta:+.6f}  iter={best_iter}  ({elapsed:.0f}s){marker}")
        sys.stdout.flush()

        results.append({"test": name, "n_features": len(cols),
                        "wape": round(wape, 6), "delta": round(delta, 6),
                        "best_iter": best_iter, "time_s": round(elapsed, 1)})

    # ── 5. Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY (sorted by WAPE, step_1 only)")
    print(f"{'='*70}")
    for r in sorted(results, key=lambda x: x["wape"]):
        d = f"  Δ={r['delta']:+.6f}" if r["test"] != "baseline" else "  (base)"
        print(f"  {r['test']:25s}  WAPE={r['wape']:.6f}{d}  ({r['n_features']} feats, {r['time_s']:.0f}s)")
    sys.stdout.flush()

    Path("exp_feature_screen_results.json").write_text(
        json.dumps({"timestamp": datetime.now().isoformat(timespec="seconds"),
                     "step": STEP_COL, "results": sorted(results, key=lambda x: x["wape"])},
                    indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved: exp_feature_screen_results.json")
    print("DONE.")


if __name__ == "__main__":
    main()
