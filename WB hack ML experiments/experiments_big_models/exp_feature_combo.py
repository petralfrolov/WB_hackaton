"""Test combinations of winning feature groups from screening."""

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

# ── Winner features ───────────────────────────────────────────────────────────

def add_rolling_cv(df):
    new = []
    for w in [4, 8, 48]:
        m, s = f"target_roll_mean_{w}", f"target_roll_std_{w}"
        if m in df.columns and s in df.columns:
            n = f"target_cv_{w}"
            df[n] = df[s] / (df[m] + 1e-6)
            new.append(n)
    return df, new

def add_fourier(df):
    new = []
    for k in [2, 3]:
        for base, period in [("hour", 24), ("day_of_week", 7)]:
            for fn, label in [(np.sin, "sin"), (np.cos, "cos")]:
                n = f"{base}_{label}_{k}"
                df[n] = fn(2 * np.pi * k * df[base] / period)
                new.append(n)
    for k in [1, 2, 3]:
        for fn, label in [(np.sin, "sin"), (np.cos, "cos")]:
            n = f"slot_{label}_{k}"
            df[n] = fn(2 * np.pi * k * df["halfhour_slot"] / 48)
            new.append(n)
    return df, new

def add_lag_ratios(df):
    new = []
    pairs = [("target_lag_1", "target_lag_48", "lag_ratio_1_48", "/"),
             ("target_lag_1", "target_lag_336", "lag_ratio_1_336", "/"),
             ("target_lag_1", "target_ema_8", "momentum_1_ema8", "-"),
             ("target_lag_1", "target_ema_24", "momentum_1_ema24", "-")]
    for a, b, n, op in pairs:
        if a in df.columns and b in df.columns:
            df[n] = df[a] / (df[b] + 1e-6) if op == "/" else df[a] - df[b]
            new.append(n)
    return df, new


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
    print("COMBINATION TEST — fourier + rolling_cv + lag_ratios")
    print("=" * 70)
    sys.stdout.flush()

    train_df, _ = load_data()
    train_df = make_features(train_df, extended=True)
    train_df = create_future_targets(train_df)

    base_cols = build_feature_cols(train_df)

    # Add all three groups
    train_df, cv_cols = add_rolling_cv(train_df)
    train_df, fourier_cols = add_fourier(train_df)
    train_df, lr_cols = add_lag_ratios(train_df)

    print(f"Base: {len(base_cols)}, +CV: {len(cv_cols)}, +Fourier: {len(fourier_cols)}, +LagRatios: {len(lr_cols)}")

    # Split once
    sup = train_df.dropna(subset=[STEP_COL])
    all_extra = cv_cols + fourier_cols + lr_cols
    keep = list(set(base_cols + all_extra + ["timestamp", STEP_COL]))
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

    combos = {
        "baseline":                 base_cols,
        "fourier":                  base_cols + fourier_cols,
        "rolling_cv":               base_cols + cv_cols,
        "lag_ratios":               base_cols + lr_cols,
        "fourier+cv":               base_cols + fourier_cols + cv_cols,
        "fourier+lr":               base_cols + fourier_cols + lr_cols,
        "cv+lr":                    base_cols + cv_cols + lr_cols,
        "fourier+cv+lr":            base_cols + fourier_cols + cv_cols + lr_cols,
    }

    results = []
    bl_wape = None
    for name, cols in combos.items():
        print(f"\n--- {name} ({len(cols)} feats) ---")
        sys.stdout.flush()
        Xf, Xv = encode_cats(fit_df[cols].copy(), val_df[cols].copy(), cols)
        t0 = time.time()
        wape, best_iter = train_eval(Xf, y_fit, Xv, y_val)
        elapsed = time.time() - t0
        if bl_wape is None:
            bl_wape = wape
        delta = wape - bl_wape
        marker = " ***" if delta < -1e-6 else ""
        print(f"  WAPE={wape:.6f}  Δ={delta:+.6f}  iter={best_iter}  ({elapsed:.0f}s){marker}")
        sys.stdout.flush()
        results.append({"test": name, "n_features": len(cols),
                        "wape": round(wape, 6), "delta": round(delta, 6),
                        "time_s": round(elapsed, 1)})

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for r in sorted(results, key=lambda x: x["wape"]):
        d = f"Δ={r['delta']:+.6f}" if r["test"] != "baseline" else "(base)"
        print(f"  {r['test']:25s}  WAPE={r['wape']:.6f}  {d}  ({r['n_features']} feats)")

    Path("exp_feature_combo_results.json").write_text(
        json.dumps({"timestamp": datetime.now().isoformat(timespec="seconds"),
                     "results": sorted(results, key=lambda x: x["wape"])},
                    indent=2), encoding="utf-8")
    print("\nSaved: exp_feature_combo_results.json\nDONE.")


if __name__ == "__main__":
    main()
