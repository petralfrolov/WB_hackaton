"""Wave 2 feature screening: more advanced ideas.

Best from wave 1: rolling_cv + lag_ratios (Δ=-0.000175).
Now test on top of that winning base.
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

# ── Wave 1 winners (always included) ─────────────────────────────────────────

def add_wave1_winners(df):
    """rolling_cv + lag_ratios."""
    new = []
    for w in [4, 8, 48]:
        m, s = f"target_roll_mean_{w}", f"target_roll_std_{w}"
        if m in df.columns and s in df.columns:
            n = f"target_cv_{w}"
            df[n] = df[s] / (df[m] + 1e-6)
            new.append(n)
    pairs = [("target_lag_1", "target_lag_48", "lag_ratio_1_48", "/"),
             ("target_lag_1", "target_lag_336", "lag_ratio_1_336", "/"),
             ("target_lag_1", "target_ema_8", "momentum_1_ema8", "-"),
             ("target_lag_1", "target_ema_24", "momentum_1_ema24", "-")]
    for a, b, n, op in pairs:
        if a in df.columns and b in df.columns:
            df[n] = df[a] / (df[b] + 1e-6) if op == "/" else df[a] - df[b]
            new.append(n)
    return df, new


# ── Wave 2 feature groups ────────────────────────────────────────────────────

def feat_cumulative_daily(df):
    """Cumulative volume shipped so far today for this route."""
    df["_day"] = df["timestamp"].dt.date
    g = df.groupby(["route_id", "_day"], sort=False)
    df["daily_cumsum"] = g[TARGET_COL].cumsum() - df[TARGET_COL]  # exclude current
    df["daily_count"] = g[TARGET_COL].cumcount()
    df.drop(columns=["_day"], inplace=True)
    return df, ["daily_cumsum", "daily_count"]

def feat_hours_since_nonzero(df):
    """Half-hours since last non-zero shipment for this route."""
    g = df.groupby("route_id", sort=False)
    nonzero = (df[TARGET_COL] > 0).astype(int)
    # cumsum to create groups of consecutive zeros
    groups = nonzero.groupby(df["route_id"], sort=False).cumsum()
    shifted_groups = g[TARGET_COL].shift(1).fillna(0)
    # Simpler: just count consecutive zeros before current
    is_zero = (g[TARGET_COL].shift(1) == 0).astype(int)
    # Use rolling trick: for each route, count consecutive zeros
    # This is a proxy: recent_zero_count
    for w in [12, 48]:
        name = f"recent_zero_count_{w}"
        df[name] = (g[TARGET_COL].shift(1) == 0).astype(float).groupby(
            df["route_id"], sort=False
        ).transform(lambda x: x.rolling(w, min_periods=1).sum())
    return df, ["recent_zero_count_12", "recent_zero_count_48"]

def feat_target_quantile_pos(df):
    """Where current lag sits vs rolling percentiles — relative position."""
    new = []
    if "target_lag_1" in df.columns:
        for w in [48, 336]:
            mean_c = f"target_roll_mean_{w}"
            min_c, max_c = f"target_roll_min_{w}", f"target_roll_max_{w}"
            # Normalized position within [min, max]
            if all(c in df.columns for c in [min_c, max_c]):
                name = f"target_qpos_{w}"
                rng = df[max_c] - df[min_c]
                df[name] = (df["target_lag_1"] - df[min_c]) / (rng + 1e-6)
                new.append(name)
    return df, new

def feat_office_cumulative(df):
    """Cumulative office-level volume today."""
    df["_day"] = df["timestamp"].dt.date
    g = df.groupby(["office_from_id", "_day"], sort=False)
    df["office_daily_cumsum"] = g[TARGET_COL].cumsum() - df[TARGET_COL]
    df.drop(columns=["_day"], inplace=True)
    return df, ["office_daily_cumsum"]

def feat_status_momentum(df):
    """Status acceleration: diff of status diffs."""
    new = []
    scols = sorted([c for c in df.columns if c.startswith("status_")
                    and not any(x in c for x in ["ratio", "lag", "roll", "sum",
                                                  "first", "last", "change"])])
    for col in scols[:4]:
        lag1 = f"{col}_lag1"
        lag2 = f"{col}_lag2"
        if lag1 in df.columns and lag2 in df.columns:
            name = f"{col}_accel"
            df[name] = (df[col] - df[lag1]) - (df[lag1] - df[lag2])
            new.append(name)
    return df, new

def feat_ema_crossover(df):
    """EMA crossover signals — fast vs slow EMA."""
    new = []
    if "target_ema_8" in df.columns and "target_ema_24" in df.columns:
        df["ema_cross_8_24"] = df["target_ema_8"] - df["target_ema_24"]
        new.append("ema_cross_8_24")
    if "target_ema_8" in df.columns and "target_ema_96" in df.columns:
        df["ema_cross_8_96"] = df["target_ema_8"] - df["target_ema_96"]
        new.append("ema_cross_8_96")
    if "target_ema_24" in df.columns and "target_ema_96" in df.columns:
        df["ema_cross_24_96"] = df["target_ema_24"] - df["target_ema_96"]
        new.append("ema_cross_24_96")
    return df, new

def feat_slot_deviation(df):
    """How much current value deviates from typical route×slot pattern."""
    new = []
    if "route_slot_mean" in df.columns and "route_slot_std" in df.columns:
        lag1 = "target_lag_1"
        if lag1 in df.columns:
            df["slot_zscore"] = (df[lag1] - df["route_slot_mean"]) / (df["route_slot_std"] + 1e-6)
            new.append("slot_zscore")
    return df, new

def feat_deconv_ratios(df):
    """Ratios between deconvolved components — flow balance."""
    new = []
    pairs = [("deconv_s_t0", "deconv_s_t1", "deconv_ratio_01"),
             ("deconv_s_t0", "deconv_s_t3", "deconv_ratio_03"),
             ("deconv_s_t0", "deconv_s_t2", "deconv_ratio_02")]
    for a, b, n in pairs:
        if a in df.columns and b in df.columns:
            df[n] = df[a] / (df[b] + 1e-6)
            new.append(n)
    # Total deconv
    dc = [c for c in ["deconv_s_t0","deconv_s_t1","deconv_s_t2","deconv_s_t3"] if c in df.columns]
    if dc:
        df["deconv_total"] = df[dc].sum(axis=1)
        df["deconv_s_t0_share"] = df["deconv_s_t0"] / (df["deconv_total"] + 1e-6) if "deconv_s_t0" in df.columns else 0
        new.extend(["deconv_total", "deconv_s_t0_share"])
    return df, new


# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_GROUPS = {
    "cumulative_daily":   feat_cumulative_daily,
    "hours_since_zero":   feat_hours_since_nonzero,
    "quantile_pos":       feat_target_quantile_pos,
    "office_cumulative":  feat_office_cumulative,
    "status_momentum":    feat_status_momentum,
    "ema_crossover":      feat_ema_crossover,
    "slot_deviation":     feat_slot_deviation,
    "deconv_ratios":      feat_deconv_ratios,
}


def encode_cats(X_fit, X_valid, feat_cols):
    cat_cols = [c for c in feat_cols if c.endswith("_id")]
    for col in cat_cols:
        cats = pd.concat([X_fit[col], X_valid[col]]).astype(str).unique()
        dtype = pd.CategoricalDtype(categories=cats)
        X_fit[col] = X_fit[col].astype(str).astype(dtype)
        X_valid[col] = X_valid[col].astype(str).astype(dtype)
    return X_fit, X_valid

def train_eval(X_fit, y_fit, X_valid, y_valid):
    m = lgb.LGBMRegressor(**FAST_LGB)
    m.fit(X_fit, y_fit, eval_set=[(X_valid, y_valid)],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    pred = m.predict(X_valid).clip(0)
    pred *= y_valid.sum() / (pred.sum() + 1e-12)
    wape = np.abs(pred - y_valid.values).sum() / y_valid.values.sum()
    return float(wape), m.best_iteration_


def main():
    print("=" * 70)
    print("WAVE 2 FEATURE SCREENING")
    print("  Base = extended + wave1 winners (cv + lag_ratios)")
    print("=" * 70)
    sys.stdout.flush()

    train_df, _ = load_data()
    train_df = make_features(train_df, extended=True)
    train_df = create_future_targets(train_df)

    base_cols = build_feature_cols(train_df)
    train_df, w1_cols = add_wave1_winners(train_df)
    enhanced_base = base_cols + w1_cols
    print(f"Enhanced base: {len(enhanced_base)} feats ({len(base_cols)} base + {len(w1_cols)} wave1)")

    # Add all wave2 groups
    group_cols = {}
    for name, fn in FEATURE_GROUPS.items():
        t0 = time.time()
        train_df, new_cols = fn(train_df)
        group_cols[name] = new_cols
        print(f"  {name}: +{len(new_cols)} cols  ({time.time()-t0:.1f}s)")
        sys.stdout.flush()

    # Split once
    sup = train_df.dropna(subset=[STEP_COL])
    all_extra = [c for cols in group_cols.values() for c in cols]
    keep = list(set(enhanced_base + all_extra + ["timestamp", STEP_COL]))
    mdf = sup[keep].copy()
    del sup, train_df

    mdf = mdf.rename(columns={"timestamp": "ts"})
    ts_max = mdf["ts"].max()
    mdf = mdf[mdf["ts"] >= ts_max - pd.Timedelta(days=TRAIN_DAYS)].sort_values("ts")
    sp = mdf["ts"].quantile(VALID_FRAC)
    fit_df = mdf[mdf["ts"] <= sp]
    val_df = mdf[mdf["ts"] > sp]
    if len(fit_df) > MAX_TRAIN_ROWS:
        fit_df = fit_df.sample(MAX_TRAIN_ROWS, random_state=RANDOM_STATE)
    y_fit, y_val = fit_df[STEP_COL], val_df[STEP_COL]
    print(f"Fit: {len(fit_df)}, Valid: {len(val_df)}")

    # Enhanced baseline (wave1 winners)
    print(f"\n{'='*50}\n  ENHANCED BASELINE ({len(enhanced_base)} feats)\n{'='*50}")
    sys.stdout.flush()
    Xf, Xv = encode_cats(fit_df[enhanced_base].copy(), val_df[enhanced_base].copy(), enhanced_base)
    t0 = time.time()
    bl_wape, bl_iter = train_eval(Xf, y_fit, Xv, y_val)
    print(f"  WAPE={bl_wape:.6f}  iter={bl_iter}  ({time.time()-t0:.0f}s)")
    sys.stdout.flush()

    results = [{"test": "enhanced_base", "n_features": len(enhanced_base),
                "wape": round(bl_wape, 6), "delta": 0.0}]

    for name, new_cols in group_cols.items():
        if not new_cols:
            continue
        cols = enhanced_base + new_cols
        print(f"\n--- {name} (+{len(new_cols)} = {len(cols)} feats) ---")
        sys.stdout.flush()
        Xf, Xv = encode_cats(fit_df[cols].copy(), val_df[cols].copy(), cols)
        t0 = time.time()
        wape, best_iter = train_eval(Xf, y_fit, Xv, y_val)
        elapsed = time.time() - t0
        delta = wape - bl_wape
        marker = " ***" if delta < -1e-6 else ""
        print(f"  WAPE={wape:.6f}  Δ={delta:+.6f}  iter={best_iter}  ({elapsed:.0f}s){marker}")
        sys.stdout.flush()
        results.append({"test": name, "n_features": len(cols),
                        "wape": round(wape, 6), "delta": round(delta, 6)})

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for r in sorted(results, key=lambda x: x["wape"]):
        d = f"Δ={r['delta']:+.6f}" if r["test"] != "enhanced_base" else "(base)"
        print(f"  {r['test']:25s}  WAPE={r['wape']:.6f}  {d}  ({r['n_features']} feats)")

    Path("exp_feature_screen2_results.json").write_text(
        json.dumps({"timestamp": datetime.now().isoformat(timespec="seconds"),
                     "results": sorted(results, key=lambda x: x["wape"])},
                    indent=2), encoding="utf-8")
    print("\nSaved: exp_feature_screen2_results.json\nDONE.")


if __name__ == "__main__":
    main()
