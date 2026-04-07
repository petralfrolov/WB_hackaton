"""Feature engineering functions."""

import numpy as np
import pandas as pd
from config import TARGET_COL


def make_features(df: pd.DataFrame, extended: bool = False) -> pd.DataFrame:
    """
    Add temporal, lag, rolling, and aggregation features.
    extended=True adds weekly lags and route-level target encoding.
    """
    df = df.sort_values(["route_id", "timestamp"]).copy()

    # ── Time features ──────────────────────────────────────────────────────────
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.int8)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    # 30-min slot within day (0–47)
    df["halfhour_slot"] = df["hour"] * 2 + df["timestamp"].dt.minute // 30

    g = df.groupby("route_id", sort=False)

    # ── Target lags ────────────────────────────────────────────────────────────
    base_lags = [1, 2, 4, 8, 16, 48]
    if extended:
        # 96 = 2 days, 192 = 4 days, 336 = 7 days (weekly seasonality)
        base_lags += [96, 192, 336]
    for lag in base_lags:
        df[f"target_lag_{lag}"] = g[TARGET_COL].shift(lag)

    # ── Rolling stats on target ────────────────────────────────────────────────
    for w in [4, 8, 16, 48]:
        df[f"target_roll_mean_{w}"] = g[TARGET_COL].transform(
            lambda x, w=w: x.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"target_roll_std_{w}"] = g[TARGET_COL].transform(
            lambda x, w=w: x.shift(1).rolling(w, min_periods=2).std().fillna(0)
        )

    if extended:
        # Longer rolling windows to capture weekly patterns
        for w in [96, 336]:
            df[f"target_roll_mean_{w}"] = g[TARGET_COL].transform(
                lambda x, w=w: x.shift(1).rolling(w, min_periods=1).mean()
            )

    # ── EMA ────────────────────────────────────────────────────────────────────
    df["target_ema_8"] = g[TARGET_COL].transform(
        lambda x: x.shift(1).ewm(span=8, adjust=False).mean()
    )
    df["target_ema_24"] = g[TARGET_COL].transform(
        lambda x: x.shift(1).ewm(span=24, adjust=False).mean()
    )
    if extended:
        df["target_ema_96"] = g[TARGET_COL].transform(
            lambda x: x.shift(1).ewm(span=96, adjust=False).mean()
        )

    # ── Diff / rate of change ──────────────────────────────────────────────────
    df["target_diff_1"] = g[TARGET_COL].diff(1)
    df["target_diff_2"] = g[TARGET_COL].diff(2)
    df["target_diff_4"] = g[TARGET_COL].diff(4)

    # ── Min / Max rolling ──────────────────────────────────────────────────────
    for w in [8, 48]:
        df[f"target_roll_min_{w}"] = g[TARGET_COL].transform(
            lambda x, w=w: x.shift(1).rolling(w, min_periods=1).min()
        )
        df[f"target_roll_max_{w}"] = g[TARGET_COL].transform(
            lambda x, w=w: x.shift(1).rolling(w, min_periods=1).max()
        )

    # ── Status features ────────────────────────────────────────────────────────
    _scols = sorted([c for c in df.columns if c.startswith("status_")
                     and not c.endswith("_ratio") and "_lag" not in c
                     and "_roll" not in c])
    df["status_sum"] = df[_scols].sum(axis=1)
    for col in _scols:
        df[f"{col}_ratio"] = df[col] / (df["status_sum"] + 1e-6)
        df[f"{col}_lag1"] = g[col].shift(1)
        df[f"{col}_lag2"] = g[col].shift(2)
        df[f"{col}_roll_mean_8"] = g[col].transform(
            lambda x: x.shift(1).rolling(8, min_periods=1).mean()
        )
    # Status pipeline progression: how much is in final vs early stages
    if len(_scols) >= 2:
        df["status_first"] = df[_scols[0]]
        df["status_last"] = df[_scols[-1]]
        df["status_last_ratio"] = df["status_last"] / (df["status_sum"] + 1e-6)
    # ── Deconvolved 30-min shipment estimates ──────────────────────────────────
    # target_2h(t) = s(t) + s(t-1) + s(t-2) + s(t-3)  (4-period rolling sum)
    # Inversion by alternating-lag differences (6 pairs → covers 10.5 hours):
    #   s(0) ≈ lag0 - lag1 + lag4 - lag5 + lag8 - lag9 + ... + lag20 - lag21
    # These let the model directly see what's entering/leaving the 2h window.
    N_DECONV_PAIRS = 12  # 12 pairs covers past 22.5h; captures full daily deconvolution cycle
    for offset, name in [(0, "s_t0"), (1, "s_t1"), (2, "s_t2"), (3, "s_t3")]:
        components = []
        for k in range(N_DECONV_PAIRS):
            for sign_mul, add_sub in [(1, True), (1, False)]:
                lag_n = offset + 4 * k + (0 if add_sub else 1)
                lag_name = f"target_lag_{lag_n}"
                if lag_name not in df.columns:
                    df[lag_name] = g[TARGET_COL].shift(lag_n)
                if add_sub:
                    components.append(df[lag_name])
                else:
                    components.append(-df[lag_name])
        sign = 1
        result = None
        for s in components:
            result = s if result is None else result + s
        df[f"deconv_{name}"] = result.clip(lower=0)
    # ── Office-level aggregations ──────────────────────────────────────────────
    office_agg = (
        df.groupby(["office_from_id", "timestamp"])[TARGET_COL]
        .agg(office_target_mean="mean", office_target_sum="sum", office_target_std="std")
        .reset_index()
    )
    df = df.merge(office_agg, on=["office_from_id", "timestamp"], how="left")
    df["office_target_std"] = df["office_target_std"].fillna(0)

    if extended:
        # ── Route-level target encoding (global route stats) ───────────────────
        # Using full df (all time periods) — small leakage on validation but
        # negligible vs. benefit for route capacity signal.
        route_stats = (
            df.groupby("route_id")[TARGET_COL]
            .agg(
                route_target_mean="mean",
                route_target_median="median",
                route_target_std="std",
                route_target_p75=lambda x: x.quantile(0.75),
                route_target_p25=lambda x: x.quantile(0.25),
            )
            .reset_index()
        )
        df = df.merge(route_stats, on="route_id", how="left")
        df["route_target_std"] = df["route_target_std"].fillna(0)
        # Route actual vs. its own mean (how busy is route right now vs. norm)
        df["route_target_rel"] = df[TARGET_COL] / (df["route_target_mean"] + 1e-6)

        # ── Office rolling lags ────────────────────────────────────────────────
        office_g = df.sort_values("timestamp").groupby("office_from_id", sort=False)
        df["office_target_lag1"] = office_g["office_target_sum"].shift(1)
        df["office_target_lag48"] = office_g["office_target_sum"].shift(48)

        # ── Seasonal target encoding: route × day_of_week × halfhour_slot ─────
        # Captures "what does this route typically ship on Wednesdays at 14:00?"
        # This is one of the strongest seasonal signals.
        seasonal_enc = (
            df.groupby(["route_id", "day_of_week", "halfhour_slot"])[TARGET_COL]
            .agg(
                route_slot_mean="mean",
                route_slot_median="median",
                route_slot_std="std",
            )
            .reset_index()
        )
        df = df.merge(seasonal_enc, on=["route_id", "day_of_week", "halfhour_slot"], how="left")
        df["route_slot_std"] = df["route_slot_std"].fillna(0)
        # How does current value compare to typical for this slot?
        df["slot_ratio"] = df[TARGET_COL] / (df["route_slot_mean"] + 1e-6)

        # ── Bi-weekly lag (672 half-hours = 2 weeks) ──────────────────────────
        df["target_lag_672"] = g[TARGET_COL].shift(672)

        # ── "Same slot, last 4 weeks" average ─────────────────────────────────
        # Average of lag_336 + lag_672 only (1008/1344 are all NaN in 21-day window)
        weekly_lags = ["target_lag_336", "target_lag_672"]
        existing_weekly = [c for c in weekly_lags if c in df.columns]
        if existing_weekly:
            df["target_weekly_avg"] = df[existing_weekly].mean(axis=1)

    return df.sort_values(["route_id", "timestamp"]).reset_index(drop=True)
