import json

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from ml_prediction import predict_lazy
from optimizer_horizons import build_plan
from schemas.optimize import OptimizeRequest, OptimizeResponse, PlanRow
from state import AppState, get_state

router = APIRouter(tags=["optimize"])


def _apply_overrides(cfg: dict, req: OptimizeRequest) -> dict:
    out = json.loads(json.dumps(cfg))
    if req.initial_stock_units is not None:
        out["initial_stock_units"] = float(req.initial_stock_units)
    if req.route_distance_km is not None:
        out["route_distance_km"] = float(req.route_distance_km)
    if req.wait_penalty_per_minute is not None:
        out["wait_penalty_per_minute"] = float(req.wait_penalty_per_minute)
    if req.underload_penalty_per_unit is not None:
        out["underload_penalty_per_unit"] = float(req.underload_penalty_per_unit)
    return out


@router.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest, state: AppState = Depends(get_state)):
    route_id = str(req.route_id)
    ts_str = req.timestamp.strftime("%Y-%m-%d %H:%M:%S")

    if route_id not in state.office_map:
        raise HTTPException(status_code=404, detail="route_id not found in train data")

    preds = predict_lazy(
        train_path=state.train_path,
        models=state.models,
        route_id=route_id,
        timestamp=ts_str,
        office_routes=state.office_routes_map.get(state.office_map.get(route_id, ""), []),
    )

    cfg = _apply_overrides(state.vehicles_cfg, req)
    init_stock = float(cfg.get("initial_stock_units", 0))
    demands = {route_id: [init_stock, preds["pred_0_2h"], preds["pred_2_4h"], preds["pred_4_6h"]]}

    plan_df = build_plan(
        timestamp=ts_str,
        demands=demands,
        vehicles_cfg=cfg,
        office_id=state.office_map.get(route_id, ""),
        incoming_vehicles=state.incoming_cfg,
    )
    plan_df["timestamp"] = pd.to_datetime(plan_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    coverage_min = (
        float((plan_df["actually_shipped"] - (plan_df["demand_new"] + plan_df["demand_carried_over"])).min())
        if not plan_df.empty else 0.0
    )

    return OptimizeResponse(
        plan=[PlanRow(**r) for r in plan_df.to_dict("records")],
        coverage_min=coverage_min,
    )
