"""Configuration and settings management router.

Handles GET/POST /config (vehicle fleet config) and PATCH /settings
(risk parameters) and GET/PUT /route-distances.
"""
import json
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends

from schemas.settings import Settings
from schemas.warehouses import RouteDistance
from core.state import AppState, get_state

router = APIRouter(tags=["settings"])

_VEHICLES_PATH = Path("data/vehicles.json")
_ROUTE_DISTANCES_PATH = Path("data/route_distances.json")


@router.get("/config")
def get_config(state: AppState = Depends(get_state)):
    """Return current vehicles configuration merged with the active confidence level."""
    return {**state.vehicles_cfg, "confidence_level": state.confidence_level}


@router.post("/config")
def set_config(payload: dict, state: AppState = Depends(get_state)):
    """Replace the entire vehicles configuration and persist it to vehicles.json."""
    state.vehicles_cfg = payload
    _VEHICLES_PATH.write_text(
        json.dumps(state.vehicles_cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"status": "ok"}


@router.patch("/settings")
def update_settings(s: Settings, state: AppState = Depends(get_state)):
    """Patch individual risk-related settings (confidence_level, route_correlation, etc.).

    Only fields present in the request body are updated.  ``confidence_level`` and
    ``route_correlation`` are also mirrored onto ``AppState`` attributes so that
    running dispatch requests pick up the new values immediately.
    """
    payload = s.model_dump(exclude_none=True)

    # confidence_level is stored separately in AppState (not in vehicles_cfg structure)
    if "confidence_level" in payload:
        cl = float(payload.pop("confidence_level"))
        state.confidence_level = max(0.0, min(1.0, cl))
        state.vehicles_cfg["confidence_level"] = state.confidence_level

    # route_correlation is stored separately in AppState
    if "route_correlation" in payload:
        rho = float(payload.pop("route_correlation"))
        state.route_correlation = max(0.0, min(1.0, rho))
        state.vehicles_cfg["route_correlation"] = state.route_correlation

    for key, val in payload.items():
        state.vehicles_cfg[key] = val
    _VEHICLES_PATH.write_text(
        json.dumps(state.vehicles_cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    changed = {k: state.vehicles_cfg[k] for k in s.model_dump(exclude_none=True) if k in state.vehicles_cfg}
    changed["confidence_level"] = state.confidence_level
    changed["route_correlation"] = state.route_correlation
    return {"status": "ok", "settings": changed}


@router.get("/route-distances", response_model=List[RouteDistance])
def get_route_distances(state: AppState = Depends(get_state)):
    """Return all route distances stored in application state."""
    return [RouteDistance(**r) for r in state.route_distances]


@router.put("/route-distances", response_model=List[RouteDistance])
def put_route_distances(items: List[RouteDistance], state: AppState = Depends(get_state)):
    """Replace all route distances and persist the new list to route_distances.json."""
    state.route_distances = [r.model_dump() for r in items]
    _ROUTE_DISTANCES_PATH.write_text(
        json.dumps({"route_distances": state.route_distances}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return items
