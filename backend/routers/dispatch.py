"""Multi-route dispatch planning router.

Handles POST /dispatch: resolves all routes for a warehouse, runs ML forecasts,
applies conformal uncertainty margins with portfolio scaling, solves the joint
IRP MILP, and returns per-route plans with aggregate coverage metrics.

Supports granularity 0.5h / 1h / 2h via deconvolution of 2h-window ML predictions.
"""
import asyncio
import copy
import json
import math
from collections import defaultdict  # noqa: F401
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from scipy.stats import norm as scipy_norm
from sqlalchemy.orm import Session

from core.conformal import get_margin
from ml.prediction import predict_lazy
from optimizer.horizons import build_plan, make_horizons
from schemas.dispatch import DispatchRequest, DispatchResponse, RouteMetrics, RoutePlan, WarehouseMetrics
from schemas.optimize import PlanRow
from core.state import AppState, get_state
from db.database import get_db
from db.queries import (
    get_warehouse_by_id,
    get_route_ids_for_warehouse,
    get_routes_for_warehouse,
    get_vehicles_cfg_for_warehouse,
    get_incoming_for_warehouse,
)
from db import models as dbm

router = APIRouter(tags=["dispatch"])

# Horizon labels in plan_df, matching build_plan output
_HORIZON_LABELS = ["A: now", "B: +2h", "C: +4h", "D: +6h"]


# ── Deconvolution helpers ─────────────────────────────────────────────────────

# Each step model predicts target_2h — the TOTAL demand within a sliding 2-hour
# window.  Step j predicts the window [(j-4)×0.5h, j×0.5h]:
#
#   step_1:  [-1.5h, +0.5h]    step_7:  [+1.5h, +3.5h]
#   step_2:  [-1.0h, +1.0h]    step_8:  [+2.0h, +4.0h]  = pred_2_4h
#   step_3:  [-0.5h, +1.5h]    step_9:  [+2.5h, +4.5h]
#   step_4:  [ 0.0h, +2.0h]    step_10: [+3.0h, +5.0h]  = pred_0_2h
#   step_5:  [+0.5h, +2.5h]    step_11: [+3.5h, +5.5h]
#   step_6:  [+1.0h, +3.0h]    step_12: [+4.0h, +6.0h]  = pred_4_6h
#
# Crucially, step_4, step_8, step_12 correspond to NON-OVERLAPPING 2h blocks:
#   step_4  = s_0 + s_1 + s_2 + s_3   (demand in [0h, 2h])
#   step_8  = s_4 + s_5 + s_6 + s_7   (demand in [2h, 4h])
#   step_12 = s_8 + s_9 + s_10 + s_11 (demand in [4h, 6h])
#
# where s_k is the demand in the half-hour slot [k×0.5h, (k+1)×0.5h].
#
# Deconvolution uses the recurrence from overlapping windows:
#   W_{j+1} − W_j = s_j − s_{j−4}
#   ⟹  s_k = (W_{k+1} − W_k) + s_{k−4}
#
# Initial seed: assume past half-hour demand is uniform within step_1's window,
# giving c = W_1 / 4 for s_{−3}, s_{−2}, s_{−1}.
#
# After computing raw s_k values, we clip negatives and rescale within each 2h
# block so that block sums exactly equal pred_0_2h, pred_2_4h, pred_4_6h.

def _deconvolve_predictions(preds: Dict[str, float], granularity: float) -> List[float]:
    """Deconvolve 2h-window ML predictions into finer granularity demand buckets.

    Args:
        preds: Dict with pred_0_2h, pred_2_4h, pred_4_6h, and optionally
               pred_step_1..pred_step_12 for shape-aware deconvolution.
        granularity: Target granularity in hours (0.5, 1.0, or 2.0).

    Returns:
        List of demand values for each future horizon bucket (excluding the
        "now" bucket which uses ready_to_ship).  Sums over each 2h block
        exactly match the corresponding 2h model prediction.
    """
    pred_0_2h = preds["pred_0_2h"]
    pred_2_4h = preds["pred_2_4h"]
    pred_4_6h = preds.get("pred_4_6h") or pred_2_4h

    if granularity == 2.0:
        return [pred_0_2h, pred_2_4h, pred_4_6h]

    # ── Collect step window predictions W[0..11] (W[i] = step_{i+1}) ─────────
    W = [None] * 12
    for i in range(12):
        key = f"pred_step_{i + 1}"
        if key in preds and preds[key] is not None:
            W[i] = max(0.0, float(preds[key]))

    # Ensure anchors are consistent with named predictions
    W[3] = pred_0_2h
    W[7] = pred_2_4h
    W[11] = pred_4_6h

    # Fill any remaining gaps via linear interpolation from anchors
    anchors = {3: pred_0_2h, 7: pred_2_4h, 11: pred_4_6h}
    for i in range(12):
        if W[i] is None:
            W[i] = _interpolate_step(i, anchors)

    # ── Recover half-hour demands s_0..s_11 via the recurrence ───────────────
    # Seed: past slots s_{-3} = s_{-2} = s_{-1} = c, where c = W[0] / 4
    c = W[0] / 4.0

    # s_0 = W[0] - 3c  (from step_1 = s_{-3} + s_{-2} + s_{-1} + s_0)
    s_raw = [0.0] * 12
    s_raw[0] = W[0] - 3.0 * c   # = c

    # For k >= 1:  s_k = (W[k] - W[k-1]) + s_{k-4}
    past = [c, c, c]  # s_{-3}, s_{-2}, s_{-1}
    for k in range(1, 12):
        s_km4 = past[k - 4 + 3] if k < 4 else s_raw[k - 4]
        s_raw[k] = (W[k] - W[k - 1]) + s_km4

    # ── Clip & rescale within each 2h block to preserve exact sums ───────────
    blocks = [(0, 4, pred_0_2h), (4, 8, pred_2_4h), (8, 12, pred_4_6h)]
    s = [0.0] * 12
    for start, end, block_total in blocks:
        raw = [max(0.0, s_raw[i]) for i in range(start, end)]
        raw_sum = sum(raw)
        if raw_sum > 0 and block_total > 0:
            scale = block_total / raw_sum
            for j, idx in enumerate(range(start, end)):
                s[idx] = raw[j] * scale
        elif block_total > 0:
            per_slot = block_total / 4.0
            for idx in range(start, end):
                s[idx] = per_slot
        # else: block_total <= 0, leave as 0

    if granularity == 0.5:
        return [round(v) for v in s]

    if granularity == 1.0:
        # Sum pairs of adjacent 0.5h slots into 1h buckets
        return [round(s[2 * k] + s[2 * k + 1]) for k in range(6)]

    # Fallback
    return [pred_0_2h, pred_2_4h, pred_4_6h]


def _interpolate_step(idx: int, known: Dict[int, float]) -> float:
    """Linearly interpolate a missing step value from known anchor points."""
    below = [(k, v) for k, v in known.items() if k <= idx]
    above = [(k, v) for k, v in known.items() if k >= idx]

    if not below and not above:
        return 0.0
    if not below:
        return above[0][1]
    if not above:
        return below[-1][1]

    k_lo, v_lo = max(below, key=lambda x: x[0])
    k_hi, v_hi = min(above, key=lambda x: x[0])

    if k_lo == k_hi:
        return v_lo

    t = (idx - k_lo) / (k_hi - k_lo)
    return v_lo + t * (v_hi - v_lo)


# ── Conformal horizon labels for allsteps NCS ────────────────────────────────

# Map granularity slots to the matching allstep horizon labels for NCS lookup.
# allstep horizons: "-1.5-0.5h", "-1-1h", "-0.5-1.5h", "0-2h", "0.5-2.5h",
#   "1-3h", "1.5-3.5h", "2-4h", "2.5-4.5h", "3-5h", "3.5-5.5h", "4-6h"
# These correspond to steps 1..12 (0-indexed: 0..11).
_ALLSTEP_HORIZON_LABELS = [
    "-1.5-0.5h", "-1-1h", "-0.5-1.5h", "0-2h",
    "0.5-2.5h", "1-3h", "1.5-3.5h", "2-4h",
    "2.5-4.5h", "3-5h", "3.5-5.5h", "4-6h",
]

def _get_conformal_margin_for_slot(
    ncs_allsteps: Dict,
    ncs_scores: Dict,
    route_id: str,
    slot_idx: int,
    granularity: float,
    alpha: float,
    pred: float,
    normalized: bool,
) -> float:
    """Get conformal margin for a deconvolved demand slot.

    For 2h granularity, uses the original 3-horizon NCS.
    For finer granularity, uses the allsteps NCS keyed by step horizon label.
    """
    if granularity == 2.0:
        horizon_map = {0: "0-2h", 1: "2-4h", 2: "4-6h"}
        h = horizon_map.get(slot_idx, "0-2h")
        return get_margin(ncs_scores, route_id, h, alpha, pred=pred, normalized=normalized)

    # For fine-grained: map slot to the corresponding step's allstep NCS
    if granularity == 0.5:
        # slot k → the deconvolution uses steps (k+3) and (k+4), use the upper step's NCS
        step_idx = min(slot_idx + 3, 11)
    elif granularity == 1.0:
        # slot k → uses steps around 2k+3, use that step's NCS
        step_idx = min(2 * slot_idx + 3, 11)
    else:
        step_idx = 3  # fallback

    h_label = _ALLSTEP_HORIZON_LABELS[step_idx]

    # Try allsteps NCS first, fallback to original NCS
    if ncs_allsteps:
        margin = get_margin(ncs_allsteps, route_id, h_label, alpha, pred=pred, normalized=True)
        if margin > 0:
            return margin

    # Fallback to nearest 2h horizon NCS
    if step_idx <= 5:
        return get_margin(ncs_scores, route_id, "0-2h", alpha, pred=pred, normalized=normalized)
    elif step_idx <= 9:
        return get_margin(ncs_scores, route_id, "2-4h", alpha, pred=pred, normalized=normalized)
    else:
        return get_margin(ncs_scores, route_id, "4-6h", alpha, pred=pred, normalized=normalized)


def _apply_portfolio_scaling(
    conformal_margins: Dict[str, List[float]],
    alpha: float,
    rho: float = 0.3,
) -> Dict[str, List[float]]:
    """Scale per-route conformal margins using portfolio diversification theory.

    Without scaling, the total fleet is sized to cover sum_i(pred_i + z*sigma_i),
    which assumes all routes peak simultaneously — a joint probability of alpha^n
    for n independent routes.  This over-reserves fleet and starves other routes.

    The portfolio correction sizes the *aggregate* buffer to achieve the desired
    joint coverage alpha while distributing the budget proportionally:

        portfolio_std_t = sqrt((1-ρ)·Σσ_i² + ρ·(Σσ_i)²)
        k_t             = portfolio_std_t / Σσ_i   ≤ 1
        margin_i_t_new  = margin_i_t · k_t

    ρ=0: routes independent — maximum diversification benefit.
    ρ=1: routes perfectly correlated — no scaling (k=1, same as before).
    Default ρ=0.3 is a conservative mid-estimate for co-located routes.

    Horizon 0 (A: now) is deterministic and always left at 0.
    """
    z_alpha = float(scipy_norm.ppf(alpha)) if 0.5 < alpha < 1.0 else None
    if z_alpha is None or z_alpha <= 0 or len(conformal_margins) <= 1:
        return conformal_margins

    route_ids = list(conformal_margins.keys())
    n_horizons = max(len(v) for v in conformal_margins.values())
    scaled = {rid: list(margins) for rid, margins in conformal_margins.items()}

    for t in range(1, n_horizons):
        raw_margins = [conformal_margins[rid][t] for rid in route_ids if t < len(conformal_margins[rid])]
        sum_m = sum(raw_margins)
        if sum_m <= 0:
            continue

        sigmas = [m / z_alpha for m in raw_margins]
        sum_var = sum(s ** 2 for s in sigmas)
        sum_sigma = sum(sigmas)

        # Portfolio standard deviation with assumed inter-route correlation ρ
        portfolio_std = math.sqrt((1.0 - rho) * sum_var + rho * sum_sigma ** 2)
        k = (z_alpha * portfolio_std) / sum_m  # ≤ 1.0

        for rid in route_ids:
            if t < len(scaled[rid]):
                scaled[rid][t] = round(scaled[rid][t] * k, 4)

    return scaled


def _aggregate_route_buckets(
    plan_df: pd.DataFrame,
    conformal_margins: Dict[str, List[float]],
    horizon_labels: List[str],
) -> tuple:
    """Aggregate per-route MILP output into totals and per-horizon buckets.

    Args:
        plan_df: Plan DataFrame produced by ``build_plan``.
        conformal_margins: Dict ``{route_id: [m0, m1, m2, m3]}`` of portfolio-scaled
            conformal margins.
        horizon_labels: Ordered list of horizon label strings (must match plan_df
            ``horizon`` column values).

    Returns:
        Tuple of:
            horizon_total_cap   — ``{label: float}`` total dispatched capacity
            horizon_total_demand — ``{label: float}`` total point-forecast demand
            horizon_total_margin — ``{label: float}`` total conformal margin (stochastic only)
            route_metrics_list   — list of per-route ``RouteMetrics`` objects
            total_capacity_sent — warehouse-wide total capacity float
            total_shipped       — warehouse-wide total shipped float
            total_cost_all      — warehouse-wide total cost float
    """
    total_capacity_sent = 0.0
    total_shipped = 0.0
    total_cost_all = 0.0
    route_metrics_list: List[RouteMetrics] = []

    horizon_total_cap: Dict[str, float]    = {h: 0.0 for h in horizon_labels}
    horizon_total_demand: Dict[str, float] = {h: 0.0 for h in horizon_labels}
    horizon_total_margin: Dict[str, float] = {h: 0.0 for h in horizon_labels}

    for rid, margins in conformal_margins.items():
        sub = plan_df[plan_df["route_id"] == rid]
        if sub.empty:
            continue

        r_cap_sent = 0.0
        r_shipped = 0.0
        r_cost = 0.0

        for i, hlabel in enumerate(horizon_labels):
            rows = sub[sub["horizon"] == hlabel]
            if rows.empty:
                continue
            first = rows.iloc[0]
            demand_pt = float(first["demand_new"])
            shipped = float(first["actually_shipped"])
            total_empty = float(rows["empty_capacity_units"].sum())
            cap_row = shipped + total_empty
            margin = margins[i] if i < len(margins) else 0.0

            r_cap_sent += cap_row
            r_shipped += shipped
            r_cost += float(rows["cost_total"].sum())

            horizon_total_cap[hlabel]    += cap_row
            horizon_total_demand[hlabel] += demand_pt
            if hlabel != "A: now":
                horizon_total_margin[hlabel] += margin

        total_capacity_sent += r_cap_sent
        total_shipped += r_shipped
        total_cost_all += r_cost

        r_fill = (r_shipped / r_cap_sent) if r_cap_sent > 0 else 0.0
        r_cpo  = (r_cost / r_shipped)    if r_shipped > 0 else 0.0
        route_metrics_list.append(RouteMetrics(
            route_id=rid,
            fill_rate=round(r_fill, 4),
            cpo=round(r_cpo, 2),
        ))

    return (
        horizon_total_cap,
        horizon_total_demand,
        horizon_total_margin,
        route_metrics_list,
        total_capacity_sent,
        total_shipped,
        total_cost_all,
    )


def _compute_p_cover_by_horizon(
    horizon_total_cap: Dict[str, float],
    horizon_total_demand: Dict[str, float],
    horizon_total_margin: Dict[str, float],
    z_alpha: float,
    horizon_labels: List[str],
) -> List[float]:
    """Compute per-horizon aggregate coverage probability using a normal approximation.

    Horizon A (``"A: now"``) is always 1.0 (deterministic init stock). For
    stochastic horizons B–D the portfolio aggregate std is recovered from the
    already-scaled margins:
        std_t = total_margin_t / z_alpha
        z_t   = (total_cap_t - total_demand_t) / std_t
        p_t   = Φ(z_t)

    When ``total_margin_t == 0`` (conformal CI disabled), falls back to a
    hard 1.0 / 0.5 rule based on whether point-forecast demand is covered.

    Args:
        horizon_total_cap: Total dispatched + empty capacity per horizon label.
        horizon_total_demand: Total point-forecast demand per horizon label.
        horizon_total_margin: Total portfolio-scaled conformal margin per horizon
            label (0.0 for ``"A: now"``).
        z_alpha: Normal quantile ``Φ⁻¹(α)``; must be positive.
        horizon_labels: Ordered list of all horizon label strings.

    Returns:
        List of probabilities aligned with ``horizon_labels``.
    """
    p_cover_by_horizon: List[float] = []
    for hlabel in horizon_labels:
        if hlabel == "A: now":
            p_cover_by_horizon.append(1.0)
            continue

        total_cap_t    = horizon_total_cap[hlabel]
        total_demand_t = horizon_total_demand[hlabel]
        total_margin_t = horizon_total_margin[hlabel]

        if total_margin_t > 0 and z_alpha > 0:
            std_t = total_margin_t / z_alpha
            z_t   = (total_cap_t - total_demand_t) / std_t
            p_t   = round(float(scipy_norm.cdf(z_t)), 4)
        else:
            p_t = 1.0 if total_cap_t > total_demand_t else 0.5

        p_cover_by_horizon.append(p_t)
    return p_cover_by_horizon


def _compute_warehouse_metrics(
    plan_df: pd.DataFrame,
    conformal_margins: Dict[str, List[float]],
    alpha: float,
    horizon_labels: List[str] = None,
    vehicles_cfg: Dict = None,
    incoming_vehicles: List[Dict] = None,
) -> WarehouseMetrics:
    """Compute p_cover, fill_rate and CPO from the solved plan.

    p_cover is the probability that the warehouse aggregate dispatch covers
    aggregate actual demand at each stochastic horizon.

    Because portfolio scaling (_apply_portfolio_scaling) provides a coverage
    guarantee at the AGGREGATE (warehouse) level — not per individual route —
    we evaluate p_cover on the sum-over-routes at each horizon:

        total_cap_t     = Σ_r  (cap sent for route r at horizon t)
        total_demand_t  = Σ_r  demand_new_{r,t}  (point forecasts, no margin)
        total_margin_t  = Σ_r  conformal_margin_{r,t}  (already portfolio-scaled)

    The aggregate std at horizon t equals total_margin_t / z_alpha, because
    _apply_portfolio_scaling was designed so that k_t * Σ raw_margins = z_alpha
    * portfolio_std_t, i.e. total_margin_t / z_alpha = portfolio_std_t exactly.

        z_t     = (total_cap_t - total_demand_t) / (total_margin_t / z_alpha)
        p_t     = Φ(z_t)

    Horizon A (t=0) is always 1.0 (init_stock is deterministic).
    p_cover = min over stochastic horizons B–D.

    Args:
        plan_df: Plan DataFrame from ``build_plan``.
        conformal_margins: Portfolio-scaled margins ``{route_id: [m0..m3]}``.
        alpha: Conformal coverage level in (0, 1).

    Returns:
        ``WarehouseMetrics`` with p_cover, p_cover_by_horizon, fill_rate, cpo,
        and per-route RouteMetrics.
    """
    z_alpha = float(scipy_norm.ppf(alpha)) if 0.0 < alpha < 1.0 else 1.645

    if horizon_labels is None:
        horizon_labels = _HORIZON_LABELS

    # Build vehicle capacity lookup from config
    veh_cap: Dict[str, float] = {}
    veh_avail: Dict[str, int] = {}
    for v in (vehicles_cfg or {}).get("vehicles", []):
        vt = v["vehicle_type"]
        veh_cap[vt] = float(v.get("capacity_units", 0))
        veh_avail[vt] = int(v.get("available", 0))

    # Fleet at last horizon (+6h): base + all incoming that arrive within the window
    last_horizon_idx = len(horizon_labels) - 1 if horizon_labels else 3
    veh_avail_at_last: Dict[str, int] = dict(veh_avail)  # copy base counts
    for iv in (incoming_vehicles or []):
        iv_vt = str(iv.get("vehicle_type", ""))
        iv_h = int(iv.get("horizon_idx", 0))
        if iv_vt in veh_avail_at_last and iv_h <= last_horizon_idx:
            veh_avail_at_last[iv_vt] = veh_avail_at_last.get(iv_vt, 0) + int(iv.get("count", 0))

    # Available fleet capacity = fleet at last horizon × capacity per type
    available_capacity_units = sum(
        veh_avail_at_last.get(vt, 0) * cap for vt, cap in veh_cap.items()
    )

    # Available fleet detail per vehicle type (show counts at +6h)
    fleet_detail: List[Dict] = []
    for v in (vehicles_cfg or {}).get("vehicles", []):
        vt = v["vehicle_type"]
        cap = float(v.get("capacity_units", 0))
        avail_base = int(v.get("available", 0))
        avail_6h = veh_avail_at_last.get(vt, avail_base)
        fleet_detail.append({
            "vehicle_type": vt,
            "available": avail_base,
            "available_at_6h": avail_6h,
            "capacity_units": cap,
            "total_capacity": round(avail_6h * cap, 1),
        })

    # Required capacity = total demand across all routes and horizons.
    # plan_df has one row per (route_id, horizon, vehicle_type) — demand_new is the
    # same for all vehicle-type rows at a given route×horizon, so we must deduplicate
    # before summing to avoid counting demand once per vehicle type.
    required_capacity_units = float(
        plan_df.drop_duplicates(subset=["route_id", "horizon"])["demand_new"].sum()
    )
    # Add conformal safety buffer (stochastic demand margin)
    total_conformal_margin = sum(
        sum(margins[1:]) for margins in conformal_margins.values()
    )
    required_capacity_with_margin = required_capacity_units + total_conformal_margin

    # Dispatched capacity = what the MILP actually allocated (may be < required)
    dispatched_mask = (plan_df["vehicle_type"] != "none") & (plan_df["vehicles_count"] > 0)
    dispatched_capacity_units = 0.0
    dispatched_detail: List[Dict] = []
    vtype_dispatch: Dict[str, Dict] = {}
    for _, row_ in plan_df[dispatched_mask].iterrows():
        vt = str(row_["vehicle_type"])
        cap = veh_cap.get(vt, 0.0)
        cnt = float(row_["vehicles_count"])
        dispatched_capacity_units += cnt * cap
        if vt not in vtype_dispatch:
            vtype_dispatch[vt] = {"vehicle_type": vt, "vehicles_count": 0.0, "capacity_units": cap, "total_capacity": 0.0}
        vtype_dispatch[vt]["vehicles_count"] += cnt
        vtype_dispatch[vt]["total_capacity"] += cnt * cap
    for vt, d in vtype_dispatch.items():
        dispatched_detail.append({
            "vehicle_type": d["vehicle_type"],
            "vehicles_count": round(d["vehicles_count"], 1),
            "capacity_units": d["capacity_units"],
            "total_capacity": round(d["total_capacity"], 1),
        })

    (
        horizon_total_cap,
        horizon_total_demand,
        horizon_total_margin,
        route_metrics_list,
        total_capacity_sent,
        total_shipped,
        total_cost_all,
    ) = _aggregate_route_buckets(plan_df, conformal_margins, horizon_labels)

    p_cover_by_horizon = _compute_p_cover_by_horizon(
        horizon_total_cap, horizon_total_demand, horizon_total_margin, z_alpha, horizon_labels
    )

    stochastic = p_cover_by_horizon[1:]
    p_cover = round(min(stochastic), 4) if stochastic else 1.0

    fill_rate = round(total_shipped / total_capacity_sent, 4) if total_capacity_sent > 0 else 0.0
    cpo = round(total_cost_all / total_shipped, 2) if total_shipped > 0 else 0.0

    fleet_utilization_ratio = (
        round(required_capacity_with_margin / available_capacity_units, 4)
        if available_capacity_units > 0 else None
    )
    fleet_capacity_shortfall = round(required_capacity_with_margin - available_capacity_units, 1)

    # ── Build detail breakdowns for each metric ──────────────────────────────
    # p_cover detail: per-horizon breakdown
    p_cover_detail: List[Dict] = []
    for i, hlabel in enumerate(horizon_labels):
        p_cover_detail.append({
            "horizon": hlabel,
            "capacity": round(horizon_total_cap.get(hlabel, 0), 1),
            "demand": round(horizon_total_demand.get(hlabel, 0), 1),
            "margin": round(horizon_total_margin.get(hlabel, 0), 1),
            "p_cover": p_cover_by_horizon[i] if i < len(p_cover_by_horizon) else 1.0,
        })

    # fill_rate detail: per-route
    fill_rate_detail: List[Dict] = []
    for rm in route_metrics_list:
        sub = plan_df[plan_df["route_id"] == rm.route_id]
        r_shipped = float(sub["actually_shipped"].sum()) if not sub.empty else 0.0
        r_cap = r_shipped + float(sub["empty_capacity_units"].sum()) if not sub.empty else 0.0
        fill_rate_detail.append({
            "route_id": rm.route_id,
            "shipped": round(r_shipped, 1),
            "capacity_sent": round(r_cap, 1),
            "fill_rate": rm.fill_rate,
        })

    # cpo detail: per-route
    cpo_detail: List[Dict] = []
    for rm in route_metrics_list:
        sub = plan_df[plan_df["route_id"] == rm.route_id]
        r_cost = float(sub["cost_total"].sum()) if not sub.empty else 0.0
        r_shipped = float(sub["actually_shipped"].sum()) if not sub.empty else 0.0
        cpo_detail.append({
            "route_id": rm.route_id,
            "cost": round(r_cost, 2),
            "shipped": round(r_shipped, 1),
            "cpo": rm.cpo,
        })

    # fleet utilization detail: demand vs available by horizon
    util_detail: List[Dict] = []
    for i, hlabel in enumerate(horizon_labels):
        demand_h = horizon_total_demand.get(hlabel, 0)
        margin_h = horizon_total_margin.get(hlabel, 0)
        cap_h = horizon_total_cap.get(hlabel, 0)
        util_detail.append({
            "horizon": hlabel,
            "demand": round(demand_h, 1),
            "margin": round(margin_h, 1),
            "demand_with_margin": round(demand_h + margin_h, 1),
            "dispatched_capacity": round(cap_h, 1),
        })

    return WarehouseMetrics(
        p_cover=p_cover,
        p_cover_by_horizon=p_cover_by_horizon,
        fill_rate=fill_rate,
        cpo=cpo,
        fleet_utilization_ratio=fleet_utilization_ratio,
        fleet_capacity_shortfall=fleet_capacity_shortfall,
        required_capacity_units=round(required_capacity_with_margin, 1),
        available_capacity_units=round(available_capacity_units, 1),
        dispatched_capacity_units=round(dispatched_capacity_units, 1),
        total_demand_units=round(required_capacity_units, 1),
        total_conformal_margin=round(total_conformal_margin, 1),
        route_metrics=route_metrics_list,
        horizon_labels=horizon_labels,
        p_cover_detail=p_cover_detail,
        fill_rate_detail=fill_rate_detail,
        cpo_detail=cpo_detail,
        fleet_detail=fleet_detail,
        dispatched_detail=dispatched_detail,
        utilization_detail=util_detail,
    )


@router.post("/dispatch", response_model=DispatchResponse)
async def dispatch(
    req: DispatchRequest,
    state: AppState = Depends(get_state),
    db: Session = Depends(get_db),
):
    """Plan joint transport dispatch for all routes of a warehouse.

    The heavy computation (ML forecasting + MILP solving) runs in a worker
    thread via ``asyncio.to_thread`` so the event loop is never blocked.
    """
    # ── 1. Resolve warehouse ─────────────────────────────────────────────────
    warehouse = get_warehouse_by_id(db, req.warehouse_id)
    if warehouse is None or warehouse.is_mock:
        raise HTTPException(status_code=404, detail="warehouse not found")

    # Global concurrency limit — non-blocking to prevent threadpool exhaustion
    if not state._dispatch_semaphore.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="Server busy — too many concurrent dispatch requests. Please retry.",
        )
    state.inc_dispatches()
    try:
        wh_lock = state.get_warehouse_lock(req.warehouse_id)
        if not wh_lock.acquire(blocking=False):
            raise HTTPException(
                status_code=409,
                detail=f"Dispatch already in progress for warehouse {req.warehouse_id}. Please wait.",
            )
        try:
            # Run blocking ML + MILP computation in a thread pool worker so the
            # asyncio event loop remains responsive for other requests.
            # SQLite is configured check_same_thread=False, so cross-thread session
            # usage is safe here (single consumer thread, no concurrent access).
            return await asyncio.to_thread(_run_dispatch, req, state, warehouse, db)
        finally:
            wh_lock.release()
    except HTTPException:
        raise
    finally:
        state.dec_dispatches()
        state._dispatch_semaphore.release()


def _run_dispatch(req: DispatchRequest, state: AppState, warehouse, db: Session) -> DispatchResponse:

    route_ids: List[str] = get_route_ids_for_warehouse(db, warehouse.id)
    office_from_id: str = warehouse.office_from_id
    ts_str = req.timestamp.strftime("%Y-%m-%d %H:%M:%S")

    # Build route metadata from DB
    db_routes = get_routes_for_warehouse(db, warehouse.id)
    route_meta = {}
    for r in db_routes:
        to_wh = get_warehouse_by_id(db, r.to_warehouse_id)
        route_meta[r.id] = {
            "id": r.id,
            "from_id": r.from_warehouse_id,
            "to_id": r.to_warehouse_id,
            "distance_km": r.distance_km,
            "ready_to_ship": r.ready_to_ship,
            "to_city": to_wh.city if to_wh else "",
        }

    # ── 2. Build config from DB, apply request-level overrides ───────────
    cfg = get_vehicles_cfg_for_warehouse(db, warehouse.id)

    if req.vehicles_override:
        cfg["vehicles"] = [v.model_dump() for v in req.vehicles_override]
    if req.wait_penalty_per_minute is not None:
        cfg["wait_penalty_per_minute"] = req.wait_penalty_per_minute

    incoming = (
        [iv.model_dump() for iv in req.incoming_vehicles]
        if req.incoming_vehicles is not None
        else get_incoming_for_warehouse(db, warehouse.id)
    )

    # ── 3. ML forecasts for all routes ───────────────────────────────────────
    granularity = req.granularity if req.granularity is not None else state.granularity
    horizons_list, _ = make_horizons(granularity)
    dyn_horizon_labels = [h[0] for h in horizons_list]

    raw_preds: Dict[str, Dict[str, float]] = {}
    demands: Dict[str, List[float]] = {}
    route_distances: Dict[str, float] = {}
    missing_routes: List[str] = []
    # All routes from this office are loaded together for correct office-level features
    office_routes = state.office_routes_map.get(office_from_id, [])

    for rid in route_ids:
        if rid not in state.office_map:
            missing_routes.append(rid)
            continue
        preds = predict_lazy(
            train_path=state.train_path,
            models=state.models,
            route_id=rid,
            timestamp=ts_str,
            office_routes=office_routes,
        )
        raw_preds[rid] = preds
        route_cfg = route_meta.get(str(rid), {})
        ready_to_ship = float(route_cfg.get("ready_to_ship", 0))

        # Deconvolve 2h predictions into finer-grained demand buckets
        deconv = _deconvolve_predictions(preds, granularity)
        demands[rid] = [ready_to_ship] + deconv
        route_distances[rid] = float(route_cfg.get("distance_km", 15.0))

    if not demands:
        raise HTTPException(
            status_code=422,
            detail=f"None of the warehouse routes found in train data. Missing: {missing_routes}",
        )

    # ── 4. Joint MILP for all routes at once ─────────────────────────────────
    alpha = req.confidence_level if req.confidence_level is not None else state.confidence_level
    normalized = state.ncs_normalized
    n_future = len(demands[next(iter(demands))]) - 1  # exclude t0

    conformal_margins: Dict[str, List[float]] = {}
    for rid in demands:
        margins = [0.0]  # t0 is deterministic
        for slot_idx in range(n_future):
            pred_val = demands[rid][slot_idx + 1]
            m = _get_conformal_margin_for_slot(
                state.ncs_allsteps, state.ncs_scores,
                rid, slot_idx, granularity, alpha, pred=pred_val, normalized=normalized,
            )
            margins.append(m)
        conformal_margins[rid] = margins

    # Portfolio scaling: reduce per-route margins so the *aggregate* buffer
    # achieves joint coverage alpha instead of inflating each route independently.
    conformal_margins = _apply_portfolio_scaling(
        conformal_margins, alpha, rho=state.route_correlation
    )
    plan_df = build_plan(
        timestamp=ts_str,
        demands=demands,
        vehicles_cfg=cfg,
        office_id=office_from_id,
        route_distances=route_distances,
        global_fleet=req.global_fleet,
        incoming_vehicles=incoming,
        conformal_margins=conformal_margins,
        granularity=granularity,
    )

    if plan_df.empty:
        raise HTTPException(status_code=500, detail="Optimizer returned an empty plan")

    plan_df["timestamp"] = pd.to_datetime(plan_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # ── 5. Split plan by route_id ─────────────────────────────────────────────
    route_plans: List[RoutePlan] = []
    total_cost = 0.0

    for rid in demands:
        sub = plan_df[plan_df["route_id"] == rid]
        if sub.empty:
            continue
        rows = [PlanRow(**r) for r in sub.to_dict("records")]
        cov = float(
            (sub["actually_shipped"] - (sub["demand_new"] + sub["demand_carried_over"])).min()
        )
        route_cost = float(sub["cost_total"].sum())
        total_cost += route_cost
        route_plans.append(RoutePlan(route_id=rid, plan=rows, coverage_min=cov))

    resp = DispatchResponse(
        warehouse_id=req.warehouse_id,
        office_from_id=office_from_id,
        timestamp=req.timestamp,
        routes=route_plans,
        total_cost=round(total_cost, 2),
        metrics=_compute_warehouse_metrics(plan_df, conformal_margins, alpha, dyn_horizon_labels, cfg, incoming),
        granularity=granularity,
        horizon_labels=dyn_horizon_labels,
    )

    # cache last dispatch in state for /call reuse (timestamp + plan)
    dispatch_snapshot = resp.model_dump()
    dispatch_snapshot["timestamp"] = ts_str
    with state._dispatch_write_lock:
        state.last_dispatch = dispatch_snapshot
        state.last_dispatch_by_warehouse[req.warehouse_id] = dispatch_snapshot

    # ── 7. Persist dispatch result to DB ─────────────────────────────────
    _persist_dispatch_result(db, req.warehouse_id, req.timestamp, granularity, dispatch_snapshot)

    return resp


def _persist_dispatch_result(
    db: Session,
    warehouse_id: str,
    timestamp: datetime,
    granularity: float,
    snapshot: dict,
) -> None:
    """Upsert dispatch result into the dispatch_results table."""
    # Determine which column to update
    col_map = {0.5: "granularity_05", 1.0: "granularity_1", 2.0: "granularity_2"}
    col_name = col_map.get(granularity, "granularity_2")

    # Make timestamp naive for SQLite storage
    ts = timestamp if timestamp.tzinfo is None else timestamp.replace(tzinfo=None)

    existing = db.query(dbm.DispatchResult).filter(
        dbm.DispatchResult.warehouse_id == warehouse_id,
        dbm.DispatchResult.timestamp == ts,
    ).first()

    serialized = json.dumps(snapshot, ensure_ascii=False, default=str)

    if existing:
        setattr(existing, col_name, serialized)
        existing.updated_at = datetime.now(timezone.utc)
    else:
        row = dbm.DispatchResult(
            warehouse_id=warehouse_id,
            timestamp=ts,
        )
        setattr(row, col_name, serialized)
        db.add(row)

    db.commit()
