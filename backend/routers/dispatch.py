"""Multi-route dispatch planning router.

Handles POST /dispatch: resolves all routes for a warehouse, runs ML forecasts,
applies conformal uncertainty margins with portfolio scaling, solves the joint
IRP MILP, and returns per-route plans with aggregate coverage metrics.
"""
import copy
import math
from collections import defaultdict  # noqa: F401
from typing import Dict, List

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from scipy.stats import norm as scipy_norm

from core.conformal import get_margin
from ml.prediction import predict_lazy
from optimizer.horizons import build_plan
from schemas.dispatch import DispatchRequest, DispatchResponse, RouteMetrics, RoutePlan, WarehouseMetrics
from schemas.optimize import PlanRow
from core.state import AppState, get_state

router = APIRouter(tags=["dispatch"])

# Horizon labels in plan_df, matching build_plan output
_HORIZON_LABELS = ["A: now", "B: +2h", "C: +4h", "D: +6h"]


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

    (
        horizon_total_cap,
        horizon_total_demand,
        horizon_total_margin,
        route_metrics_list,
        total_capacity_sent,
        total_shipped,
        total_cost_all,
    ) = _aggregate_route_buckets(plan_df, conformal_margins, _HORIZON_LABELS)

    p_cover_by_horizon = _compute_p_cover_by_horizon(
        horizon_total_cap, horizon_total_demand, horizon_total_margin, z_alpha, _HORIZON_LABELS
    )

    stochastic = p_cover_by_horizon[1:]
    p_cover = round(min(stochastic), 4) if stochastic else 1.0

    fill_rate = round(total_shipped / total_capacity_sent, 4) if total_capacity_sent > 0 else 0.0
    cpo = round(total_cost_all / total_shipped, 2) if total_shipped > 0 else 0.0

    return WarehouseMetrics(
        p_cover=p_cover,
        p_cover_by_horizon=p_cover_by_horizon,
        fill_rate=fill_rate,
        cpo=cpo,
        route_metrics=route_metrics_list,
    )


@router.post("/dispatch", response_model=DispatchResponse)
def dispatch(req: DispatchRequest, state: AppState = Depends(get_state)):
    """Plan joint transport dispatch for all routes of a warehouse.

    Runs the full pipeline: ML demand forecast → conformal margins →
    portfolio scaling → joint MILP → coverage metrics.

    Args:
        req: Dispatch request containing warehouse_id, timestamp, and optional
            overrides for vehicles, confidence level, and incoming vehicles.
        state: Application state injected by FastAPI.

    Returns:
        ``DispatchResponse`` with per-route plans, total cost, and aggregate
        warehouse metrics (p_cover, fill_rate, CPO).

    Raises:
        HTTPException 404: Warehouse not found.
        HTTPException 422: No routes from this warehouse found in train data.
        HTTPException 500: MILP optimizer returned an empty plan.
    """
    # ── 1. Resolve warehouse ─────────────────────────────────────────────────
    warehouse = next((w for w in state.warehouses if w["id"] == req.warehouse_id), None)
    if warehouse is None:
        raise HTTPException(status_code=404, detail="warehouse not found")

    route_ids: List[str] = warehouse["route_ids"]
    office_from_id: str = warehouse["office_from_id"]
    ts_str = req.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    route_meta = {str(route["id"]): route for route in state.route_distances}

    # ── 2. Build config, apply request-level overrides ───────────────────────
    cfg = copy.deepcopy(state.vehicles_cfg)

    if req.vehicles_override:
        cfg["vehicles"] = [v.model_dump() for v in req.vehicles_override]
    if req.wait_penalty_per_minute is not None:
        cfg["wait_penalty_per_minute"] = req.wait_penalty_per_minute

    incoming = (
        [iv.model_dump() for iv in req.incoming_vehicles]
        if req.incoming_vehicles is not None
        else state.incoming_cfg
    )

    # ── 3. ML forecasts for all routes ───────────────────────────────────────
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
        route_cfg = route_meta.get(str(rid), {})
        ready_to_ship = float(route_cfg.get("ready_to_ship", 0))
        demands[rid] = [
            ready_to_ship,
            preds["pred_0_2h"],
            preds["pred_2_4h"],
            preds["pred_4_6h"],
        ]
        route_distances[rid] = float(route_cfg.get("distance_km", 15.0))

    if not demands:
        raise HTTPException(
            status_code=422,
            detail=f"None of the warehouse routes found in train data. Missing: {missing_routes}",
        )

    # ── 4. Joint MILP for all routes at once ─────────────────────────────────
    alpha = req.confidence_level if req.confidence_level is not None else state.confidence_level
    normalized = state.ncs_normalized
    conformal_margins = {
        rid: [
            0.0,  # t0: init_stock is deterministic, no uncertainty
            get_margin(state.ncs_scores, rid, "0-2h", alpha, pred=demands[rid][1], normalized=normalized),
            get_margin(state.ncs_scores, rid, "2-4h", alpha, pred=demands[rid][2], normalized=normalized),
            get_margin(state.ncs_scores, rid, "4-6h", alpha, pred=demands[rid][3], normalized=normalized),
        ]
        for rid in demands
    }
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
        metrics=_compute_warehouse_metrics(plan_df, conformal_margins, alpha),
    )

    # cache last dispatch in state for /call reuse (timestamp + plan)
    state.last_dispatch = resp.model_dump()
    state.last_dispatch["timestamp"] = ts_str
    state.last_dispatch_by_warehouse[req.warehouse_id] = state.last_dispatch
    return resp
