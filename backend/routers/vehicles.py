"""Fleet management router — CRUD for vehicle types and incoming vehicle records.

Vehicle types are global definitions; availability is per-warehouse via warehouse_vehicles.
"""
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from schemas.vehicles import IncomingVehicle, IncomingVehicleList, Vehicle, VehicleUpdate
from db.database import get_db
from db import models as m
from db.queries import (
    get_all_vehicle_types,
    get_vehicle_type_by_name,
    get_fleet_for_warehouse,
    get_all_warehouses,
)

router = APIRouter(tags=["vehicles"])


# ── Available fleet ──────────────────────────────────────────────────────────

@router.get("/vehicles", response_model=List[Vehicle])
async def list_vehicles(
    warehouse_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """Return vehicle types. If warehouse_id provided, includes per-warehouse availability."""
    if warehouse_id:
        fleet = get_fleet_for_warehouse(db, warehouse_id)
        return [Vehicle(**v, warehouse_id=warehouse_id) for v in fleet]
    # No warehouse_id: return all vehicle types with available=0 (global definitions)
    vtypes = get_all_vehicle_types(db)
    result = []
    for vt in vtypes:
        # Pick representative availability from first warehouse that has it
        wv = db.query(m.WarehouseVehicle).filter(m.WarehouseVehicle.vehicle_type_id == vt.id).first()
        result.append(Vehicle(
            vehicle_type=vt.vehicle_type,
            capacity_units=vt.capacity_units,
            cost_per_km=vt.cost_per_km,
            available=wv.available if wv else 0,
            underload_penalty=vt.underload_penalty,
            fixed_dispatch_cost=vt.fixed_dispatch_cost,
        ))
    return result


@router.post("/vehicles", response_model=Vehicle, status_code=201)
async def add_vehicle(v: Vehicle, db: Session = Depends(get_db)):
    """Add a new vehicle type globally and create warehouse_vehicle rows for all real warehouses."""
    existing = get_vehicle_type_by_name(db, v.vehicle_type)
    if existing:
        raise HTTPException(status_code=400, detail="vehicle_type already exists")
    vt = m.VehicleType(
        vehicle_type=v.vehicle_type,
        capacity_units=v.capacity_units,
        cost_per_km=v.cost_per_km,
        underload_penalty=v.underload_penalty or 0.0,
        fixed_dispatch_cost=v.fixed_dispatch_cost or 0.0,
    )
    db.add(vt)
    db.flush()
    # Create per-warehouse entries for all real warehouses
    warehouses = get_all_warehouses(db, include_mock=False)
    for wh in warehouses:
        db.add(m.WarehouseVehicle(
            warehouse_id=wh.id,
            vehicle_type_id=vt.id,
            available=v.available,
        ))
    db.commit()
    return v


@router.patch("/vehicles/{vehicle_type}", response_model=Vehicle)
async def update_vehicle(
    vehicle_type: str,
    v: VehicleUpdate,
    db: Session = Depends(get_db),
):
    """Update a vehicle type's parameters. If warehouse_id in body, update availability for that warehouse only."""
    vt = get_vehicle_type_by_name(db, vehicle_type)
    if vt is None:
        raise HTTPException(status_code=404, detail="vehicle_type not found")
    updates = v.model_dump(exclude_none=True)
    wh_id = updates.pop("warehouse_id", None)
    # Update global vehicle type fields
    for field in ("capacity_units", "cost_per_km", "underload_penalty", "fixed_dispatch_cost"):
        if field in updates:
            setattr(vt, field, updates[field])
    if "vehicle_type" in updates and updates["vehicle_type"] != vehicle_type:
        vt.vehicle_type = updates["vehicle_type"]
    # Update availability per-warehouse or globally
    if "available" in updates:
        if wh_id:
            wv = db.query(m.WarehouseVehicle).filter(
                m.WarehouseVehicle.warehouse_id == wh_id,
                m.WarehouseVehicle.vehicle_type_id == vt.id,
            ).first()
            if wv:
                wv.available = updates["available"]
            else:
                db.add(m.WarehouseVehicle(warehouse_id=wh_id, vehicle_type_id=vt.id, available=updates["available"]))
        else:
            # Update all warehouses
            db.query(m.WarehouseVehicle).filter(
                m.WarehouseVehicle.vehicle_type_id == vt.id
            ).update({"available": updates["available"]})
    db.commit()
    # Return updated vehicle
    wv = db.query(m.WarehouseVehicle).filter(m.WarehouseVehicle.vehicle_type_id == vt.id).first()
    return Vehicle(
        vehicle_type=vt.vehicle_type,
        capacity_units=vt.capacity_units,
        cost_per_km=vt.cost_per_km,
        available=wv.available if wv else 0,
        underload_penalty=vt.underload_penalty,
        fixed_dispatch_cost=vt.fixed_dispatch_cost,
        warehouse_id=wh_id,
    )


@router.delete("/vehicles/{vehicle_type}", status_code=200)
async def delete_vehicle(vehicle_type: str, db: Session = Depends(get_db)):
    """Remove a vehicle type and all associated warehouse_vehicle and incoming records."""
    vt = get_vehicle_type_by_name(db, vehicle_type)
    if vt is None:
        raise HTTPException(status_code=404, detail="vehicle_type not found")
    db.delete(vt)  # cascades to warehouse_vehicles and incoming_vehicles
    db.commit()
    return {"status": "ok"}


@router.post("/vehicles/{vehicle_type}/sync", status_code=200)
async def sync_vehicle_across_warehouses(
    vehicle_type: str,
    warehouse_id: str = Query(..., description="Source warehouse whose availability to copy"),
    db: Session = Depends(get_db),
):
    """Copy availability AND incoming vehicles of a vehicle type from source warehouse to all other real warehouses."""
    vt = get_vehicle_type_by_name(db, vehicle_type)
    if vt is None:
        raise HTTPException(status_code=404, detail="vehicle_type not found")
    source = db.query(m.WarehouseVehicle).filter(
        m.WarehouseVehicle.warehouse_id == warehouse_id,
        m.WarehouseVehicle.vehicle_type_id == vt.id,
    ).first()
    if source is None:
        raise HTTPException(status_code=404, detail="vehicle not found for source warehouse")
    # Get incoming records for this vehicle type from source warehouse
    source_incoming = db.query(m.IncomingVehicle).filter(
        m.IncomingVehicle.warehouse_id == warehouse_id,
        m.IncomingVehicle.vehicle_type_id == vt.id,
    ).all()
    all_wh = get_all_warehouses(db, include_mock=False)
    for wh in all_wh:
        if wh.id == warehouse_id:
            continue
        # Sync base availability
        wv = db.query(m.WarehouseVehicle).filter(
            m.WarehouseVehicle.warehouse_id == wh.id,
            m.WarehouseVehicle.vehicle_type_id == vt.id,
        ).first()
        if wv:
            wv.available = source.available
        else:
            db.add(m.WarehouseVehicle(warehouse_id=wh.id, vehicle_type_id=vt.id, available=source.available))
        # Sync incoming vehicles: delete existing for this type, copy from source
        db.query(m.IncomingVehicle).filter(
            m.IncomingVehicle.warehouse_id == wh.id,
            m.IncomingVehicle.vehicle_type_id == vt.id,
        ).delete()
        for si in source_incoming:
            db.add(m.IncomingVehicle(
                warehouse_id=wh.id,
                vehicle_type_id=vt.id,
                horizon_idx=si.horizon_idx,
                count=si.count,
            ))
    db.commit()
    return {"status": "ok", "synced_available": source.available}


# ── Incoming vehicles ────────────────────────────────────────────────────────

@router.get("/incoming-vehicles", response_model=IncomingVehicleList)
async def get_incoming(
    warehouse_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """Return incoming vehicle records. If warehouse_id provided, scoped to that warehouse."""
    q = db.query(m.IncomingVehicle, m.VehicleType).join(
        m.VehicleType, m.IncomingVehicle.vehicle_type_id == m.VehicleType.id
    )
    if warehouse_id:
        q = q.filter(m.IncomingVehicle.warehouse_id == warehouse_id)
    else:
        # Return first warehouse's incoming as representative (backward compat)
        first_wh = db.query(m.Warehouse).filter(m.Warehouse.is_mock.is_(False)).first()
        if first_wh:
            q = q.filter(m.IncomingVehicle.warehouse_id == first_wh.id)
    rows = q.all()
    incoming = [
        IncomingVehicle(
            horizon_idx=iv.horizon_idx,
            vehicle_type=vt.vehicle_type,
            count=iv.count,
            warehouse_id=iv.warehouse_id,
        )
        for iv, vt in rows
    ]
    return IncomingVehicleList(incoming=incoming)


@router.put("/incoming-vehicles", response_model=IncomingVehicleList)
async def put_incoming(
    payload: IncomingVehicleList,
    warehouse_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """Replace incoming vehicles list. If warehouse_id provided, replace for that warehouse only.
    Otherwise replace for all real warehouses uniformly."""
    target_ids = []
    if warehouse_id:
        target_ids = [warehouse_id]
    else:
        warehouses = get_all_warehouses(db, include_mock=False)
        target_ids = [wh.id for wh in warehouses]

    for wh_id in target_ids:
        # Delete existing
        db.query(m.IncomingVehicle).filter(m.IncomingVehicle.warehouse_id == wh_id).delete()
        # Insert new
        for item in payload.incoming:
            vt = get_vehicle_type_by_name(db, item.vehicle_type)
            if vt is None:
                continue
            db.add(m.IncomingVehicle(
                warehouse_id=wh_id,
                horizon_idx=item.horizon_idx,
                vehicle_type_id=vt.id,
                count=item.count,
            ))
    db.commit()
    return payload


@router.post("/incoming-vehicles", response_model=IncomingVehicle, status_code=201)
async def add_incoming(item: IncomingVehicle, db: Session = Depends(get_db)):
    """Append an incoming-vehicle record for all real warehouses (or specific if warehouse_id set)."""
    vt = get_vehicle_type_by_name(db, item.vehicle_type)
    if vt is None:
        raise HTTPException(status_code=400, detail="unknown vehicle_type")
    target_ids = []
    if item.warehouse_id:
        target_ids = [item.warehouse_id]
    else:
        warehouses = get_all_warehouses(db, include_mock=False)
        target_ids = [wh.id for wh in warehouses]
    for wh_id in target_ids:
        db.add(m.IncomingVehicle(
            warehouse_id=wh_id,
            horizon_idx=item.horizon_idx,
            vehicle_type_id=vt.id,
            count=item.count,
        ))
    db.commit()
    return item


@router.patch("/incoming-vehicles/{idx}", response_model=IncomingVehicle)
async def update_incoming(
    idx: int,
    item: IncomingVehicle,
    warehouse_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """Update an incoming-vehicle record by list index."""
    q = db.query(m.IncomingVehicle)
    if warehouse_id:
        q = q.filter(m.IncomingVehicle.warehouse_id == warehouse_id)
    else:
        first_wh = db.query(m.Warehouse).filter(m.Warehouse.is_mock.is_(False)).first()
        if first_wh:
            q = q.filter(m.IncomingVehicle.warehouse_id == first_wh.id)
    records = q.all()
    if not (0 <= idx < len(records)):
        raise HTTPException(status_code=404, detail="incoming index out of range")
    vt = get_vehicle_type_by_name(db, item.vehicle_type)
    if vt is None:
        raise HTTPException(status_code=400, detail="unknown vehicle_type")
    rec = records[idx]
    rec.horizon_idx = item.horizon_idx
    rec.vehicle_type_id = vt.id
    rec.count = item.count
    db.commit()
    return item


@router.delete("/incoming-vehicles/{idx}", status_code=200)
async def delete_incoming(
    idx: int,
    warehouse_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """Remove an incoming-vehicle record by list index."""
    q = db.query(m.IncomingVehicle)
    if warehouse_id:
        q = q.filter(m.IncomingVehicle.warehouse_id == warehouse_id)
    else:
        first_wh = db.query(m.Warehouse).filter(m.Warehouse.is_mock.is_(False)).first()
        if first_wh:
            q = q.filter(m.IncomingVehicle.warehouse_id == first_wh.id)
    records = q.all()
    if not (0 <= idx < len(records)):
        raise HTTPException(status_code=404, detail="incoming index out of range")
    db.delete(records[idx])
    db.commit()
    return {"status": "ok"}
