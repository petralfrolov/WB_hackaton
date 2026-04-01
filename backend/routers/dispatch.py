import copy
from collections import defaultdict
from typing import Dict, List

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from ml_prediction import predict_lazy
from optimizer_horizons import build_plan
from schemas.dispatch import DispatchRequest, DispatchResponse, RoutePlan
from schemas.optimize import PlanRow
from state import AppState, get_state

router = APIRouter(tags=["dispatch"])


@router.post("/dispatch", response_model=DispatchResponse)
def dispatch(req: DispatchRequest, state: AppState = Depends(get_state)):
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
    plan_df = build_plan(
        timestamp=ts_str,
        demands=demands,
        vehicles_cfg=cfg,
        office_id=office_from_id,
        route_distances=route_distances,
        global_fleet=req.global_fleet,
        incoming_vehicles=incoming,
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
    )

    # cache last dispatch in state for /call reuse (timestamp + plan)
    state.last_dispatch = resp.model_dump()
    state.last_dispatch["timestamp"] = ts_str
    return resp
