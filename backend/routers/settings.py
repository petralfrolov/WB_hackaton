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
    payload = s.model_dump(exclude_none=True)

    # Пер-категорийные штрафы
    state.vehicles_cfg.setdefault("underload_penalty_per_unit_by_cat", {})
    cat_map = state.vehicles_cfg["underload_penalty_per_unit_by_cat"]

    if "underload_penalty_per_unit_by_cat" in payload:
        incoming_map = payload.pop("underload_penalty_per_unit_by_cat") or {}
        for cat, val in incoming_map.items():
            cat_map[cat] = float(val)

    for cat_field, cat_name in [
        ("underload_penalty_compact", "compact"),
        ("underload_penalty_mid", "mid"),
        ("underload_penalty_large", "large"),
    ]:
        if cat_field in payload:
            cat_map[cat_name] = float(payload.pop(cat_field))

    for key, val in payload.items():
        state.vehicles_cfg[key] = val
    _VEHICLES_PATH.write_text(
        json.dumps(state.vehicles_cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    changed = {k: state.vehicles_cfg[k] for k in s.model_dump(exclude_none=True) if k in state.vehicles_cfg}
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
