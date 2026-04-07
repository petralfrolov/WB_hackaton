"""Fast feature hypothesis testing framework.

Uses 1 seed, 1000 estimators for quick iteration (~3-4 min per run).
Tests feature groups against baseline to find what actually helps.
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
from config import TRACK, FUTURE_TARGET_COLS, TARGET_COL, FORECAST_POINTS
from data import load_data, create_future_targets, build_feature_cols, split_data, encode_categoricals
from features import make_features
from metrics import WapePlusRbias

# ── Fast LGB params ───────────────────────────────────────────────────────────
FAST_LGB = dict(
    n_estimators=1500, learning_rate=0.05, num_leaves=63,
    min_child_samples=30, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    objective="quantile", alpha=0.55,
    n_jobs=-1, verbose=-1, random_state=42,
)

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE GROUPS — each function takes df and returns df with new columns
# ═══════════════════════════════════════════════════════════════════════════════

def feat_rolling_cv(df):
    """Rolling coefficient of variation (std/mean)."""
    for w in [4, 8, 48]:
        mean_col, std_col = f"target_roll_mean_{w}", f"target_roll_std_{w}"
        if mean_col in df.columns and std_col in df.columns:
            df[f"target_cv_{w}"] = df[std_col] / (df[mean_col] + 1e-6)
    return df

def feat_fourier_harmonics(df):
    """Higher Fourier harmonics for hour, dow, halfhour_slot."""
    for k in [2, 3]:
        df[f"hour_sin_{k}"] = np.sin(2 * np.pi * k * df["hour"] / 24)
        df[f"hour_cos_{k}"] = np.cos(2 * np.pi * k * df["hour"] / 24)
        df[f"dow_sin_{k}"] = np.sin(2 * np.pi * k * df["day_of_week"] / 7)
        df[f"dow_cos_{k}"] = np.cos(2 * np.pi * k * df["day_of_week"] / 7)
    for k in [1, 2, 3]:
        df[f"slot_sin_{k}"] = np.sin(2 * np.pi * k * df["halfhour_slot"] / 48)
        df[f"slot_cos_{k}"] = np.cos(2 * np.pi * k * df["halfhour_slot"] / 48)
    return df

def feat_dest_warehouse_agg(df):
    """Cross-route destination warehouse aggregates."""
    if "office_to_id" in df.columns:
        dest_agg = (
            df.groupby(["office_to_id", "timestamp"])[TARGET_COL]
            .agg(dest_target_mean="mean", dest_target_sum="sum")
            .reset_index()
        )
        df = df.merge(dest_agg, on=["office_to_id", "timestamp"], how="left")
    return df

def feat_lag_ratios(df):
    """Lag ratios and momentum signals."""
    if "target_lag_1" in df.columns and "target_lag_48" in df.columns:
        df["lag_ratio_1_48"] = df["target_lag_1"] / (df["target_lag_48"] + 1e-6)
    if "target_lag_1" in df.columns and "target_lag_336" in df.columns:
        df["lag_ratio_1_336"] = df["target_lag_1"] / (df["target_lag_336"] + 1e-6)
    if "target_lag_1" in df.columns and "target_ema_8" in df.columns:
        df["momentum_1_ema8"] = df["target_lag_1"] - df["target_ema_8"]
    if "target_lag_1" in df.columns and "target_ema_24" in df.columns:
        df["momentum_1_ema24"] = df["target_lag_1"] - df["target_ema_24"]
    return df

def feat_time_interactions(df):
    """Continuous time-of-day and dow×time interaction."""
    df["time_of_day"] = df["hour"] + df["timestamp"].dt.minute / 60.0
    df["dow_x_time"] = df["day_of_week"] * 48 + df["halfhour_slot"]
    return df

def feat_route_office_rank(df):
    """Route busyness rank within office (percentile)."""
    if "route_target_mean" in df.columns:
        df["route_office_rank"] = (
            df.groupby("office_from_id")["route_target_mean"].rank(pct=True)
        )
    return df

def feat_zero_share(df):
    """Fraction of zeros in recent history per route — sparse demand indicator."""
    g = df.groupby("route_id", sort=False)
    for w in [48, 336]:
        df[f"zero_share_{w}"] = g[TARGET_COL].transform(
            lambda x, w=w: (x.shift(1).rolling(w, min_periods=1) == 0).mean()
        )
    return df

def feat_recent_max_ratio(df):
    """Current lag vs recent max — captures demand spikes."""
    if "target_lag_1" in df.columns:
        for w in [8, 48]:
            max_col = f"target_roll_max_{w}"
            if max_col in df.columns:
                df[f"lag1_to_max_{w}"] = df["target_lag_1"] / (df[max_col] + 1e-6)
    return df

def feat_status_change_rate(df):
    """How fast pipeline status changes — velocity of pipeline processing."""
    _scols = sorted([c for c in df.columns if c.startswith("status_")
                     and not c.endswith("_ratio") and "_lag" not in c
                     and "_roll" not in c])
    g = df.groupby("route_id", sort=False)
    for col in _scols[:3]:  # only first 3 to limit feature count
        if f"{col}_lag1" in df.columns:
            df[f"{col}_change"] = df[col] - df[f"{col}_lag1"]
    return df

def feat_target_range_ratio(df):
    """Rolling max-min range / mean — measures variability shape."""
    for w in [8, 48]:
        max_col, min_col, mean_col = f"target_roll_max_{w}", f"target_roll_min_{w}", f"target_roll_mean_{w}"
        if all(c in df.columns for c in [max_col, min_col, mean_col]):
            df[f"target_range_ratio_{w}"] = (df[max_col] - df[min_col]) / (df[mean_col] + 1e-6)
    return df

def feat_weekly_diff(df):
    """Difference vs same-time last week — weekly trend."""
    if "target_lag_1" in df.columns and "target_lag_336" in df.columns:
        df["weekly_diff"] = df["target_lag_1"] - df["target_lag_336"]
    if "target_lag_1" in df.columns and "target_lag_672" in df.columns:
        df["biweekly_diff"] = df["target_lag_1"] - df["target_lag_672"]
    return df

def feat_office_route_ratio(df):
    """Route's share of its office total — structural demand share."""
    if "target_lag_1" in df.columns and "office_target_sum" in df.columns:
        g = df.groupby("route_id", sort=False)
        office_g = df.groupby("office_from_id", sort=False)
        # Lagged version to avoid leakage
        df["office_lag1_sum"] = office_g["office_target_sum"].shift(1)
        df["route_share_office"] = df["target_lag_1"] / (df["office_lag1_sum"] + 1e-6)
    return df

def feat_peak_indicator(df):
    """Binary/soft indicators for peak hours and weekday patterns."""
    h = df["hour"]
    df["is_morning_peak"] = ((h >= 8) & (h <= 11)).astype(np.int8)
    df["is_evening_peak"] = ((h >= 16) & (h <= 19)).astype(np.int8)
    df["is_night"] = ((h >= 22) | (h <= 5)).astype(np.int8)
    df["is_weekday_daytime"] = ((df["is_weekend"] == 0) & (h >= 8) & (h <= 18)).astype(np.int8)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_GROUPS = {
    "baseline":           [],                    # no extra features
    "rolling_cv":         [feat_rolling_cv],
    "fourier":            [feat_fourier_harmonics],
    "dest_warehouse":     [feat_dest_warehouse_agg],
    "lag_ratios":         [feat_lag_ratios],
    "time_interact":      [feat_time_interactions],
    "route_rank":         [feat_route_office_rank],
    "zero_share":         [feat_zero_share],
    "recent_max_ratio":   [feat_recent_max_ratio],
    "status_change":      [feat_status_change_rate],
    "range_ratio":        [feat_target_range_ratio],
    "weekly_diff":        [feat_weekly_diff],
    "office_route_ratio": [feat_office_route_ratio],
    "peak_indicator":     [feat_peak_indicator],
}

# Which tests to run (modify this list to select)
TESTS_TO_RUN = [
    "baseline",
    "rolling_cv",
    "fourier",
    "dest_warehouse",
    "lag_ratios",
    "time_interact",
    "route_rank",
    "zero_share",
    "recent_max_ratio",
    "status_change",
    "range_ratio",
    "weekly_diff",
    "office_route_ratio",
    "peak_indicator",
]


def run_test(test_name, feature_funcs, train_feat_df):
    """Run a single feature group test, return metrics dict."""
    t0 = time.time()
    df = train_feat_df.copy()

    # Apply feature functions
    for fn in feature_funcs:
        df = fn(df)

    feat_cols = build_feature_cols(df)
    X_fit, y_fit, X_valid, y_valid, _, _, _ = split_data(df, feat_cols)
    X_fit, X_valid, _, cat_features = encode_categoricals(X_fit, X_valid, X_fit[:0], feat_cols)

    # Train single-seed model
    params = {**FAST_LGB}
    models = {}
    for step_col in FUTURE_TARGET_COLS:
        m = lgb.LGBMRegressor(**params)
        m.fit(
            X_fit, y_fit[step_col],
            eval_set=[(X_valid, y_valid[step_col])],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        models[step_col] = m
        vs = m.best_score_.get("valid_0", {})
        score = next(iter(vs.values()), float("nan"))
        print(f"    {step_col:20s}  best_iter={m.best_iteration_:4d}  score={score:.4f}")
        sys.stdout.flush()

    # Predict on validation
    preds = pd.DataFrame(index=X_valid.index)
    for col, mdl in models.items():
        preds[col] = mdl.predict(X_valid).clip(0)

    # Per-step calibration
    metric = WapePlusRbias()
    for col in FUTURE_TARGET_COLS:
        true_sum = y_valid[col].sum()
        pred_sum = preds[col].sum()
        f = true_sum / pred_sum if pred_sum > 0 else 1.0
        preds[col] = (preds[col] * f).clip(lower=0)

    w, r, t = metric.calculate_components(y_valid, preds)
    elapsed = time.time() - t0

    result = {
        "test": test_name,
        "n_features": len(feat_cols),
        "wape": round(w, 6),
        "rbias": round(r, 6),
        "total": round(t, 6),
        "time_s": round(elapsed, 1),
    }
    return result


def main():
    print("=" * 70)
    print("FAST FEATURE HYPOTHESIS TESTING")
    print("  1 seed | 1500 trees | num_leaves=63 | alpha=0.55")
    print("=" * 70)
    sys.stdout.flush()

    # Load and build base features once
    print("\nLoading data...")
    sys.stdout.flush()
    train_df, test_df = load_data()

    print("Building base features (extended + deconv)...")
    sys.stdout.flush()
    train_df = make_features(train_df, extended=True)

    print("Creating future targets...")
    sys.stdout.flush()
    train_df = create_future_targets(train_df)

    results = []

    for test_name in TESTS_TO_RUN:
        feature_funcs = FEATURE_GROUPS[test_name]
        print(f"\n{'='*50}")
        print(f"  TEST: {test_name}  ({len(feature_funcs)} feature funcs)")
        print(f"{'='*50}")
        sys.stdout.flush()

        result = run_test(test_name, feature_funcs, train_df)
        results.append(result)

        print(f"\n  >> {test_name}: WAPE={result['wape']:.6f}  "
              f"RBias={result['rbias']:.6f}  Total={result['total']:.6f}  "
              f"({result['n_features']} feats, {result['time_s']:.0f}s)")
        sys.stdout.flush()

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: Feature Group Results (sorted by Total)")
    print("=" * 70)
    results_sorted = sorted(results, key=lambda x: x["total"])
    baseline_total = next((r["total"] for r in results if r["test"] == "baseline"), None)
    for r in results_sorted:
        delta = f"  Δ={r['total'] - baseline_total:+.6f}" if baseline_total else ""
        print(f"  {r['test']:25s}  Total={r['total']:.6f}  WAPE={r['wape']:.6f}  "
              f"feats={r['n_features']:3d}  {r['time_s']:5.0f}s{delta}")
    sys.stdout.flush()

    # Log results
    log_path = Path("feature_test_results.json")
    log_data = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "lgb_params": FAST_LGB,
        "results": results_sorted,
    }
    log_path.write_text(json.dumps(log_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved results to {log_path}")
    print("DONE.")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
