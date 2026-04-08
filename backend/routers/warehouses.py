"""Warehouse information router.

Handles GET /warehouses and GET /warehouses/{id}: returns warehouse metadata
with current aggregated ready-to-ship quantities from the routes table.
"""
from typing import Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from schemas.warehouses import WarehouseInfo
from db.database import get_db
from db.queries import (
    get_all_warehouses,
    get_warehouse_by_id,
    get_route_ids_for_warehouse,
    get_ready_to_ship_for_warehouse,
)
from core.state import get_state

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


# ── Sankey (goods status flow) ────────────────────────────────────────────────

_STATUS_COLS = [f"status_{i}" for i in range(1, 9)]
_STATUS_LABELS = [
    "Принят",
    "Сортировка",
    "Упаковка",
    "Контроль",
    "Стеллаж",
    "Перемещение",
    "Ожидание",
    "Группировка",
]


@router.get("/warehouses/{warehouse_id}/sankey")
async def get_sankey(
    warehouse_id: str,
    timestamp: Optional[str] = Query(None, description="ISO timestamp, e.g. 2024-01-15T10:00"),
    route_id: Optional[str] = Query(None, description="Specific route id; omit for aggregate"),
    db: Session = Depends(get_db),
):
    """Return Sankey diagram data (goods status flow) from training parquet.

    Reads ``status_1``–``status_8`` columns for the warehouse routes at the
    closest available timestamp and returns ``{nodes, links}`` in the same
    format the front-end ``SankeyData`` expects.
    """
    wh = get_warehouse_by_id(db, warehouse_id)
    if wh is None or wh.is_mock:
        raise HTTPException(status_code=404, detail="warehouse not found")

    state = get_state()
    train_path = state.train_path
    if not train_path.exists():
        raise HTTPException(status_code=503, detail="training data not available")

    route_ids: List[str] = (
        [route_id] if route_id else get_route_ids_for_warehouse(db, warehouse_id)
    )
    if not route_ids:
        raise HTTPException(status_code=404, detail="no routes for warehouse")

    # Read only needed columns with predicate pushdown on route_id
    int_ids = []
    for rid in route_ids:
        try:
            int_ids.append(int(rid))
        except (ValueError, TypeError):
            pass
    if not int_ids:
        raise HTTPException(status_code=404, detail="no numeric route IDs")

    filters = [("route_id", "in", int_ids)]
    columns = ["route_id", "timestamp"] + _STATUS_COLS
    try:
        df = pd.read_parquet(train_path, columns=columns, filters=filters)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"parquet read error: {exc}")

    if df.empty:
        raise HTTPException(status_code=404, detail="no data for routes")

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Pick the closest timestamp
    if timestamp:
        target = pd.Timestamp(timestamp)
    else:
        target = df["timestamp"].max()

    # Find the closest available timestamp
    unique_ts = df["timestamp"].unique()
    idx = abs(unique_ts - target).argmin()
    closest_ts = unique_ts[idx]
    snap = df[df["timestamp"] == closest_ts]

    # Sum status columns across all selected routes
    totals = snap[_STATUS_COLS].sum()

    # Build Sankey nodes & links
    vals = [int(totals[f"status_{i+1}"]) for i in range(8)]
    nodes = [
        {"id": f"status_{i+1}", "label": _STATUS_LABELS[i], "value": vals[i]}
        for i in range(8)
    ]
    links = []
    for i in range(7):
        # Link value = min of adjacent nodes so the band never exceeds either node
        link_val = min(vals[i], vals[i + 1]) if vals[i] > 0 and vals[i + 1] > 0 else max(vals[i], vals[i + 1])
        links.append({
            "source": f"status_{i+1}",
            "target": f"status_{i+2}",
            "value": max(link_val, 1),  # d3-sankey ignores zero-value links
        })

    return {
        "nodes": nodes,
        "links": links,
        "timestamp": str(closest_ts),
    }
