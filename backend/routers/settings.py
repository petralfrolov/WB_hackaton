"""Configuration and settings management router.

Handles GET/POST /config (vehicle fleet config) and PATCH /settings
(risk parameters) and GET/PUT /route-distances.
"""
from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from schemas.settings import Settings
from schemas.warehouses import RouteDistance
from core.state import AppState, get_state
from db.database import get_db
from db.queries import (
    get_all_settings,
    get_all_vehicle_types,
    get_fleet_for_warehouse,
    get_all_warehouses,
    get_route_distances_list,
    get_vehicle_type_by_name,
    set_setting,
    route_to_api_dict,
    vtype_to_dict,
)
from db import models as dbm

router = APIRouter(tags=["settings"])


@router.get("/config")
async def get_config(
    state: AppState = Depends(get_state),
    db: Session = Depends(get_db),
):
    """Return current vehicles configuration merged with the active confidence level."""
    settings = get_all_settings(db)
    # For backward compat: return a vehicles_cfg-like dict
    # Use first real warehouse's fleet as representative
    warehouses = get_all_warehouses(db, include_mock=False)
    vehicles = []
    if warehouses:
        vehicles = get_fleet_for_warehouse(db, warehouses[0].id)
    return {
        "vehicles": vehicles,
        "wait_penalty_per_minute": settings.get("wait_penalty_per_minute", 8.0),
        "economy_threshold": settings.get("economy_threshold", 0.0),
        "confidence_level": state.confidence_level,
        "granularity": state.granularity,
    }


@router.post("/config")
async def set_config(
    payload: dict,
    state: AppState = Depends(get_state),
    db: Session = Depends(get_db),
):
    """Replace the entire vehicles configuration."""
    # Update vehicle types and availability
    if "vehicles" in payload:
        for v_data in payload["vehicles"]:
            vt = get_vehicle_type_by_name(db, v_data["vehicle_type"])
            if vt is None:
                vt = dbm.VehicleType(
                    vehicle_type=v_data["vehicle_type"],
                    capacity_units=v_data.get("capacity_units", 0),
                    cost_per_km=v_data.get("cost_per_km", 0),
                    underload_penalty=v_data.get("underload_penalty", 0),
                    fixed_dispatch_cost=v_data.get("fixed_dispatch_cost", 0),
                )
                db.add(vt)
                db.flush()
            else:
                vt.capacity_units = v_data.get("capacity_units", vt.capacity_units)
                vt.cost_per_km = v_data.get("cost_per_km", vt.cost_per_km)
                vt.underload_penalty = v_data.get("underload_penalty", vt.underload_penalty)
                vt.fixed_dispatch_cost = v_data.get("fixed_dispatch_cost", vt.fixed_dispatch_cost)
            # Update availability for all real warehouses
            if "available" in v_data:
                warehouses = get_all_warehouses(db, include_mock=False)
                for wh in warehouses:
                    wv = db.query(dbm.WarehouseVehicle).filter(
                        dbm.WarehouseVehicle.warehouse_id == wh.id,
                        dbm.WarehouseVehicle.vehicle_type_id == vt.id,
                    ).first()
                    if wv:
                        wv.available = v_data["available"]
                    else:
                        db.add(dbm.WarehouseVehicle(
                            warehouse_id=wh.id,
                            vehicle_type_id=vt.id,
                            available=v_data["available"],
                        ))
    # Update settings
    for key in ("wait_penalty_per_minute", "economy_threshold", "confidence_level", "granularity"):
        if key in payload:
            set_setting(db, key, payload[key])
    if "confidence_level" in payload:
        state.confidence_level = float(payload["confidence_level"])
    if "granularity" in payload:
        state.granularity = float(payload["granularity"])
    db.commit()
    return {"status": "ok"}


@router.patch("/settings")
async def update_settings(
    s: Settings,
    state: AppState = Depends(get_state),
    db: Session = Depends(get_db),
):
    """Patch individual risk-related settings."""
    payload = s.model_dump(exclude_none=True)

    if "confidence_level" in payload:
        cl = float(payload["confidence_level"])
        state.confidence_level = max(0.0, min(1.0, cl))
        set_setting(db, "confidence_level", state.confidence_level)

    if "route_correlation" in payload:
        rho = float(payload["route_correlation"])
        state.route_correlation = max(0.0, min(1.0, rho))

    if "granularity" in payload:
        g = float(payload["granularity"])
        if g in (0.5, 1.0, 2.0):
            state.granularity = g
            set_setting(db, "granularity", state.granularity)

    if "wait_penalty_per_minute" in payload:
        set_setting(db, "wait_penalty_per_minute", payload["wait_penalty_per_minute"])

    if "economy_threshold" in payload:
        set_setting(db, "economy_threshold", payload["economy_threshold"])

    db.commit()

    changed = {k: v for k, v in payload.items()}
    changed["confidence_level"] = state.confidence_level
    changed["route_correlation"] = state.route_correlation
    changed["granularity"] = state.granularity
    return {"status": "ok", "settings": changed}


@router.get("/route-distances", response_model=List[RouteDistance])
async def get_route_distances(db: Session = Depends(get_db)):
    """Return all route distances."""
    routes = get_route_distances_list(db)
    return [RouteDistance(**r) for r in routes]


@router.put("/route-distances", response_model=List[RouteDistance])
async def put_route_distances(items: List[RouteDistance], db: Session = Depends(get_db)):
    """Replace all route distances."""
    for item in items:
        route = db.query(dbm.Route).filter(dbm.Route.id == item.id).first()
        if route:
            route.distance_km = item.distance_km
            route.ready_to_ship = item.ready_to_ship
        # Note: from_id/to_id/from_city/to_city changes are ignored (FK-based)
    db.commit()
    return items
