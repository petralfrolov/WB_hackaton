"""Warehouse information router.

Handles GET /warehouses and GET /warehouses/{id}: returns warehouse metadata
with current aggregated ready-to-ship quantities from the routes table.
"""
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from schemas.warehouses import WarehouseInfo
from db.database import get_db
from db.queries import (
    get_all_warehouses,
    get_warehouse_by_id,
    get_route_ids_for_warehouse,
    get_ready_to_ship_for_warehouse,
)

router = APIRouter(tags=["warehouses"])


def _warehouse_to_info(wh, db: Session) -> WarehouseInfo:
    return WarehouseInfo(
        id=wh.id,
        name=wh.name,
        city=wh.city,
        lat=wh.lat,
        lng=wh.lng,
        office_from_id=wh.office_from_id,
        route_ids=get_route_ids_for_warehouse(db, wh.id),
        ready_to_ship=get_ready_to_ship_for_warehouse(db, wh.id),
    )


@router.get("/warehouses", response_model=List[WarehouseInfo])
async def list_warehouses(db: Session = Depends(get_db)):
    """Return all real warehouses with their current ready-to-ship totals."""
    warehouses = get_all_warehouses(db, include_mock=False)
    return [_warehouse_to_info(wh, db) for wh in warehouses]


@router.get("/warehouses/{warehouse_id}", response_model=WarehouseInfo)
async def get_warehouse(warehouse_id: str, db: Session = Depends(get_db)):
    """Return a single warehouse by ID with its ready-to-ship total."""
    wh = get_warehouse_by_id(db, warehouse_id)
    if wh is None or wh.is_mock:
        raise HTTPException(status_code=404, detail="warehouse not found")
    return _warehouse_to_info(wh, db)
