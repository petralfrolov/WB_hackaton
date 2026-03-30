import json
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends

from schemas.settings import Settings
from schemas.warehouses import RouteDistance
from state import AppState, get_state

router = APIRouter(tags=["settings"])

_VEHICLES_PATH = Path("vehicles.json")
_ROUTE_DISTANCES_PATH = Path("route_distances.json")


@router.get("/config")
def get_config(state: AppState = Depends(get_state)):
    return state.vehicles_cfg


@router.post("/config")
def set_config(payload: dict, state: AppState = Depends(get_state)):
    state.vehicles_cfg = payload
    _VEHICLES_PATH.write_text(
        json.dumps(state.vehicles_cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"status": "ok"}


@router.patch("/settings")
def update_settings(s: Settings, state: AppState = Depends(get_state)):
    for key, val in s.model_dump(exclude_none=True).items():
        state.vehicles_cfg[key] = val
    _VEHICLES_PATH.write_text(
        json.dumps(state.vehicles_cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    changed = {k: state.vehicles_cfg[k] for k in s.model_dump(exclude_none=True)}
    return {"status": "ok", "settings": changed}


@router.get("/route-distances", response_model=List[RouteDistance])
def get_route_distances(state: AppState = Depends(get_state)):
    return [RouteDistance(**r) for r in state.route_distances]


@router.put("/route-distances", response_model=List[RouteDistance])
def put_route_distances(items: List[RouteDistance], state: AppState = Depends(get_state)):
    state.route_distances = [r.model_dump() for r in items]
    _ROUTE_DISTANCES_PATH.write_text(
        json.dumps({"route_distances": state.route_distances}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return items
