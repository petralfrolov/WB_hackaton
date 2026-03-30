import json
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from schemas.vehicles import IncomingVehicle, IncomingVehicleList, Vehicle
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
def update_vehicle(vehicle_type: str, v: Vehicle, state: AppState = Depends(get_state)):
    vehicles = state.vehicles_cfg.get("vehicles", [])
    idx = next((i for i, x in enumerate(vehicles) if x["vehicle_type"] == vehicle_type), None)
    if idx is None:
        raise HTTPException(status_code=404, detail="vehicle_type not found")
    vehicles[idx] = v.model_dump()
    _VEHICLES_PATH.write_text(
        json.dumps(state.vehicles_cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return v


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


@router.put("/incoming-vehicles", response_model=IncomingVehicleList)
def set_incoming(payload: IncomingVehicleList, state: AppState = Depends(get_state)):
    state.incoming_cfg = [iv.model_dump() for iv in payload.incoming]
    _INCOMING_PATH.write_text(
        json.dumps({"incoming": state.incoming_cfg}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload
