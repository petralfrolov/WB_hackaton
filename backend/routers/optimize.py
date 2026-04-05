"""Single-route transport optimisation router.

Handles POST /optimize: a lightweight wrapper around the full dispatch pipeline
scoped to one route_id. Useful for per-route what-if analysis.
"""
import json

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from core.conformal import get_margin
from ml.prediction import predict_lazy
from optimizer.horizons import build_plan
from schemas.optimize import OptimizeRequest, OptimizeResponse, PlanRow
from core.state import AppState, get_state

router = APIRouter(tags=["optimize"])


def _apply_overrides(cfg: dict, req: OptimizeRequest) -> dict:
    out = json.loads(json.dumps(cfg))
    if req.wait_penalty_per_minute is not None:
        out["wait_penalty_per_minute"] = float(req.wait_penalty_per_minute)
    return out


@router.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest, state: AppState = Depends(get_state)):
    """Optimise transport dispatch for a single route.

    Convenience wrapper around the full dispatch pipeline scoped to one route.
    Uses ``predict_lazy`` for on-demand ML forecasting and ``build_plan`` for
    the MILP solution.

    Args:
        req: Optimization request with route_id, timestamp, and optional overrides.
        state: Application state injected by FastAPI.

    Returns:
        ``OptimizeResponse`` with a per-horizon plan and the minimum coverage slack.

    Raises:
        HTTPException 404: route_id not found in training data.
    """
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
    route_meta = next((route for route in state.route_distances if str(route["id"]) == route_id), {})
    init_stock = float(route_meta.get("ready_to_ship", 0))
    route_distance = float(route_meta.get("distance_km", 15.0))
    demands = {route_id: [round(init_stock), preds["pred_0_2h"], preds["pred_2_4h"], preds["pred_4_6h"]]}

    alpha = req.confidence_level if req.confidence_level is not None else state.confidence_level
    normalized = state.ncs_normalized
    conformal_margins = {
        route_id: [
            0.0,  # t0: init_stock is deterministic, no uncertainty
            get_margin(state.ncs_scores, route_id, "0-2h", alpha, pred=preds["pred_0_2h"], normalized=normalized),
            get_margin(state.ncs_scores, route_id, "2-4h", alpha, pred=preds["pred_2_4h"], normalized=normalized),
            get_margin(state.ncs_scores, route_id, "4-6h", alpha, pred=preds["pred_4_6h"], normalized=normalized),
        ]
    }
    plan_df = build_plan(
        timestamp=ts_str,
        demands=demands,
        vehicles_cfg=cfg,
        office_id=state.office_map.get(route_id, ""),
        route_distances={route_id: route_distance},
        incoming_vehicles=state.incoming_cfg,
        conformal_margins=conformal_margins,
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
