import json
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from schemas.vehicles import IncomingVehicle, IncomingVehicleList, Vehicle, VehicleUpdate
from state import AppState, get_state

router = APIRouter(tags=["vehicles"])

_VEHICLES_PATH = Path("vehicles.json")
_INCOMING_PATH = Path("incoming_vehicles.json")


# ── Available fleet ──────────────────────────────────────────────────────────

@router.get("/vehicles", response_model=List[Vehicle])
def list_vehicles(state: AppState = Depends(get_state)):
    return [Vehicle(**v) for v in state.vehicles_cfg.get("vehicles", [])]


@router.post("/vehicles", response_model=Vehicle, status_code=201)
def add_vehicle(v: Vehicle, state: AppState = Depends(get_state)):
    vehicles = state.vehicles_cfg.setdefault("vehicles", [])
    if any(x["vehicle_type"] == v.vehicle_type for x in vehicles):
        raise HTTPException(status_code=400, detail="vehicle_type already exists")
    vehicles.append(v.model_dump())
    _VEHICLES_PATH.write_text(
        json.dumps(state.vehicles_cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return v


@router.patch("/vehicles/{vehicle_type}", response_model=Vehicle)
def update_vehicle(vehicle_type: str, v: VehicleUpdate, state: AppState = Depends(get_state)):
    vehicles = state.vehicles_cfg.get("vehicles", [])
    idx = next((i for i, x in enumerate(vehicles) if x["vehicle_type"] == vehicle_type), None)
    if idx is None:
        raise HTTPException(status_code=404, detail="vehicle_type not found")
    # merge partial fields
    updated = {**vehicles[idx], **{k: val for k, val in v.model_dump(exclude_none=True).items()}}
    vehicles[idx] = updated
    _VEHICLES_PATH.write_text(
        json.dumps(state.vehicles_cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return Vehicle(**vehicles[idx])


@router.delete("/vehicles/{vehicle_type}", status_code=200)
def delete_vehicle(vehicle_type: str, state: AppState = Depends(get_state)):
    vehicles = state.vehicles_cfg.get("vehicles", [])
    new_list = [x for x in vehicles if x["vehicle_type"] != vehicle_type]
    if len(new_list) == len(vehicles):
        raise HTTPException(status_code=404, detail="vehicle_type not found")
    state.vehicles_cfg["vehicles"] = new_list
    _VEHICLES_PATH.write_text(
        json.dumps(state.vehicles_cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"status": "ok"}


# ── Incoming vehicles ────────────────────────────────────────────────────────

@router.get("/incoming-vehicles", response_model=IncomingVehicleList)
def get_incoming(state: AppState = Depends(get_state)):
    return IncomingVehicleList(incoming=state.incoming_cfg)


@router.post("/incoming-vehicles", response_model=IncomingVehicle, status_code=201)
def add_incoming(item: IncomingVehicle, state: AppState = Depends(get_state)):
    state.incoming_cfg.append(item.model_dump())
    _INCOMING_PATH.write_text(
        json.dumps({"incoming": state.incoming_cfg}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return item


@router.patch("/incoming-vehicles/{idx}", response_model=IncomingVehicle)
def update_incoming(idx: int, item: IncomingVehicle, state: AppState = Depends(get_state)):
    if not (0 <= idx < len(state.incoming_cfg)):
        raise HTTPException(status_code=404, detail="incoming index out of range")
    state.incoming_cfg[idx] = item.model_dump()
    _INCOMING_PATH.write_text(
        json.dumps({"incoming": state.incoming_cfg}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return item


@router.delete("/incoming-vehicles/{idx}", status_code=200)
def delete_incoming(idx: int, state: AppState = Depends(get_state)):
    if not (0 <= idx < len(state.incoming_cfg)):
        raise HTTPException(status_code=404, detail="incoming index out of range")
    state.incoming_cfg.pop(idx)
    _INCOMING_PATH.write_text(
        json.dumps({"incoming": state.incoming_cfg}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"status": "ok"}
