"""Fleet management router — CRUD for vehicle types and incoming vehicle records.
"""
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from schemas.vehicles import IncomingVehicle, IncomingVehicleList, Vehicle, VehicleUpdate
from core.state import AppState, get_state, atomic_write_json

router = APIRouter(tags=["vehicles"])

_VEHICLES_PATH = Path("data/vehicles.json")
_INCOMING_PATH = Path("data/incoming_vehicles.json")


# ── Available fleet ──────────────────────────────────────────────────────────

@router.get("/vehicles", response_model=List[Vehicle])
async def list_vehicles(state: AppState = Depends(get_state)):
    """Return all configured vehicle types."""
    return [Vehicle(**v) for v in state.vehicles_cfg.get("vehicles", [])]


@router.post("/vehicles", response_model=Vehicle, status_code=201)
async def add_vehicle(v: Vehicle, state: AppState = Depends(get_state)):
    """Add a new vehicle type and persist to vehicles.json.

    Raises:
        HTTPException 400: A vehicle with this vehicle_type already exists.
    """
    vehicles = state.vehicles_cfg.setdefault("vehicles", [])
    if any(x["vehicle_type"] == v.vehicle_type for x in vehicles):
        raise HTTPException(status_code=400, detail="vehicle_type already exists")
    vehicles.append(v.model_dump())
    atomic_write_json(_VEHICLES_PATH, state.vehicles_cfg)
    return v


@router.patch("/vehicles/{vehicle_type}", response_model=Vehicle)
async def update_vehicle(vehicle_type: str, v: VehicleUpdate, state: AppState = Depends(get_state)):
    """Partially update a vehicle type's parameters and persist to vehicles.json.

    Raises:
        HTTPException 404: vehicle_type not found.
    """
    vehicles = state.vehicles_cfg.get("vehicles", [])
    idx = next((i for i, x in enumerate(vehicles) if x["vehicle_type"] == vehicle_type), None)
    if idx is None:
        raise HTTPException(status_code=404, detail="vehicle_type not found")
    # merge partial fields
    updated = {**vehicles[idx], **{k: val for k, val in v.model_dump(exclude_none=True).items()}}
    vehicles[idx] = updated
    atomic_write_json(_VEHICLES_PATH, state.vehicles_cfg)
    return Vehicle(**vehicles[idx])


@router.delete("/vehicles/{vehicle_type}", status_code=200)
async def delete_vehicle(vehicle_type: str, state: AppState = Depends(get_state)):
    """Remove a vehicle type and persist the change to vehicles.json.

    Raises:
        HTTPException 404: vehicle_type not found.
    """
    vehicles = state.vehicles_cfg.get("vehicles", [])
    new_list = [x for x in vehicles if x["vehicle_type"] != vehicle_type]
    if len(new_list) == len(vehicles):
        raise HTTPException(status_code=404, detail="vehicle_type not found")
    state.vehicles_cfg["vehicles"] = new_list
    atomic_write_json(_VEHICLES_PATH, state.vehicles_cfg)
    return {"status": "ok"}


# ── Incoming vehicles ────────────────────────────────────────────────────────

@router.get("/incoming-vehicles", response_model=IncomingVehicleList)
async def get_incoming(state: AppState = Depends(get_state)):
    """Return the full list of incoming vehicle records."""
    return IncomingVehicleList(incoming=state.incoming_cfg)


@router.post("/incoming-vehicles", response_model=IncomingVehicle, status_code=201)
async def add_incoming(item: IncomingVehicle, state: AppState = Depends(get_state)):
    """Append an incoming-vehicle record and persist to incoming_vehicles.json."""
    state.incoming_cfg.append(item.model_dump())
    atomic_write_json(_INCOMING_PATH, {"incoming": state.incoming_cfg})
    return item


@router.patch("/incoming-vehicles/{idx}", response_model=IncomingVehicle)
async def update_incoming(idx: int, item: IncomingVehicle, state: AppState = Depends(get_state)):
    """Update an incoming-vehicle record by list index and persist.

    Raises:
        HTTPException 404: Index out of range.
    """
    if not (0 <= idx < len(state.incoming_cfg)):
        raise HTTPException(status_code=404, detail="incoming index out of range")
    state.incoming_cfg[idx] = item.model_dump()
    atomic_write_json(_INCOMING_PATH, {"incoming": state.incoming_cfg})
    return item


@router.delete("/incoming-vehicles/{idx}", status_code=200)
async def delete_incoming(idx: int, state: AppState = Depends(get_state)):
    """Remove an incoming-vehicle record by list index and persist.

    Raises:
        HTTPException 404: Index out of range.
    """
    if not (0 <= idx < len(state.incoming_cfg)):
        raise HTTPException(status_code=404, detail="incoming index out of range")
    state.incoming_cfg.pop(idx)
    atomic_write_json(_INCOMING_PATH, {"incoming": state.incoming_cfg})
    return {"status": "ok"}


@router.put("/incoming-vehicles", response_model=IncomingVehicleList)
async def put_incoming(payload: IncomingVehicleList, state: AppState = Depends(get_state)):
    """Replace entire incoming vehicles list."""
    state.incoming_cfg = [item.model_dump() for item in payload.incoming]
    atomic_write_json(_INCOMING_PATH, {"incoming": state.incoming_cfg})
    return IncomingVehicleList(incoming=state.incoming_cfg)
