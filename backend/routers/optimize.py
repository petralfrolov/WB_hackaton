"""Single-route transport optimisation router.

Handles POST /optimize: a lightweight wrapper around the full dispatch pipeline
scoped to one route_id, useful for per-route what-if analysis.
"""
import asyncio
import json
from typing import Dict, List

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ml.prediction import predict_lazy
from routers.dispatch import _deconvolve_predictions, _get_conformal_margin_for_slot
from optimizer.horizons import build_plan
from schemas.optimize import OptimizeRequest, OptimizeResponse, PlanRow
from core.state import AppState, get_state
from db.database import get_db
from db import models as dbm
from db.queries import get_vehicles_cfg_for_warehouse, get_incoming_for_warehouse

router = APIRouter(tags=["optimize"])


def _apply_overrides(cfg: dict, req: OptimizeRequest) -> dict:
    out = json.loads(json.dumps(cfg))
    if req.wait_penalty_per_minute is not None:
        out["wait_penalty_per_minute"] = float(req.wait_penalty_per_minute)
    return out


def _run_optimize(
    req: OptimizeRequest,
    state: AppState,
    cfg: dict,
    incoming: list,
    init_stock: float,
    route_distance: float,
    office_id: str,
) -> OptimizeResponse:
    """Blocking computation kernel: ML forecast + MILP solve for one route.

    Runs in a thread pool worker (called via ``asyncio.to_thread``) so the
    event loop is never blocked during the heavy computation.
    """
    route_id = str(req.route_id)
    ts_str = req.timestamp.strftime("%Y-%m-%d %H:%M:%S")

    preds = predict_lazy(
        train_path=state.train_path,
        models=state.models,
        route_id=route_id,
        timestamp=ts_str,
        office_routes=state.office_routes_map.get(office_id, []),
    )

    granularity = state.granularity
    deconv = _deconvolve_predictions(preds, granularity)
    demands = {route_id: [init_stock] + deconv}

    alpha = req.confidence_level if req.confidence_level is not None else state.confidence_level
    normalized = state.ncs_normalized
    n_future = len(deconv)
    conformal_margins = {
        route_id: [0.0] + [
            _get_conformal_margin_for_slot(
                state.ncs_allsteps, state.ncs_scores,
                route_id, slot_idx, granularity, alpha,
                pred=float(deconv[slot_idx]), normalized=normalized,
            )
            for slot_idx in range(n_future)
        ]
    }

    plan_df = build_plan(
        timestamp=ts_str,
        demands=demands,
        vehicles_cfg=cfg,
        office_id=office_id,
        route_distances={route_id: route_distance},
        incoming_vehicles=incoming,
        conformal_margins=conformal_margins,
        granularity=granularity,
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


@router.post("/optimize", response_model=OptimizeResponse)
async def optimize(
    req: OptimizeRequest,
    state: AppState = Depends(get_state),
    db: Session = Depends(get_db),
):
    """Optimise transport dispatch for a single route.

    Convenience wrapper around the full dispatch pipeline scoped to one route.
    Uses ``predict_lazy`` for on-demand ML forecasting and ``build_plan`` for
    the MILP solution.  The heavy computation runs in a worker thread via
    ``asyncio.to_thread`` so the event loop is never blocked.

    Args:
        req: Optimization request with route_id, timestamp, and optional overrides.
        state: Application state injected by FastAPI.
        db: Database session injected by FastAPI.

    Returns:
        ``OptimizeResponse`` with a per-horizon plan and the minimum coverage slack.

    Raises:
        HTTPException 404: route_id not found in training data or database.
    """
    route_id = str(req.route_id)

    if route_id not in state.office_map:
        raise HTTPException(status_code=404, detail="route_id not found in train data")

    # Look up route in DB to obtain vehicle config, route metadata, and incoming vehicles.
    route = db.query(dbm.Route).filter(dbm.Route.id == route_id).first()
    if route is None:
        raise HTTPException(status_code=404, detail="route_id not found in database")

    office_id = state.office_map.get(route_id, "")
    cfg = _apply_overrides(get_vehicles_cfg_for_warehouse(db, route.from_warehouse_id), req)
    incoming = get_incoming_for_warehouse(db, route.from_warehouse_id)
    init_stock = float(route.ready_to_ship)
    route_distance = float(route.distance_km)

    # Run blocking ML + MILP computation in a thread pool worker.
    # SQLite is configured check_same_thread=False, but we intentionally read
    # all DB data above (in the async context) and pass plain Python objects to
    # the thread to avoid any cross-thread session access.
    return await asyncio.to_thread(
        _run_optimize, req, state, cfg, incoming, init_stock, route_distance, office_id
    )
