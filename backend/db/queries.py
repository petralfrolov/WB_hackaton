"""Database query helpers shared across routers."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from config import DEFAULT_ECONOMY_THRESHOLD, DEFAULT_TRAIN_PATH, DEFAULT_WAIT_PENALTY_PER_MINUTE

from .models import (
    DispatchResult,
    IncomingVehicle,
    Route,
    Setting,
    VehicleType,
    Warehouse,
    WarehouseVehicle,
)

logger = logging.getLogger(__name__)


# ── Warehouses ────────────────────────────────────────────────────────────────

def get_all_warehouses(db: Session, include_mock: bool = False) -> List[Warehouse]:
    q = db.query(Warehouse)
    if not include_mock:
        q = q.filter(Warehouse.is_mock.is_(False))
    return q.all()


def get_warehouse_by_id(db: Session, warehouse_id: str) -> Optional[Warehouse]:
    return db.query(Warehouse).filter(Warehouse.id == warehouse_id).first()


def get_route_ids_for_warehouse(db: Session, warehouse_id: str) -> List[str]:
    rows = db.query(Route.id).filter(Route.from_warehouse_id == warehouse_id).all()
    return [r[0] for r in rows]


def get_ready_to_ship_for_warehouse(db: Session, warehouse_id: str) -> int:
    result = (
        db.query(func.coalesce(func.sum(Route.ready_to_ship), 0))
        .filter(Route.from_warehouse_id == warehouse_id)
        .scalar()
    )
    return int(result)


# ── Routes ─────────────────────────────────────────────────────────────────────

def get_all_routes(db: Session) -> List[Route]:
    return db.query(Route).all()


def get_routes_for_warehouse(db: Session, warehouse_id: str) -> List[Route]:
    return db.query(Route).filter(Route.from_warehouse_id == warehouse_id).all()


def route_to_api_dict(route: Route, db: Session) -> Dict[str, Any]:
    """Convert a Route ORM object to the API-compatible dict with from_city/to_city."""
    from_wh = get_warehouse_by_id(db, route.from_warehouse_id)
    to_wh = get_warehouse_by_id(db, route.to_warehouse_id)
    return {
        "id": route.id,
        "from_id": route.from_warehouse_id,
        "to_id": route.to_warehouse_id,
        "from_city": from_wh.city if from_wh else "",
        "to_city": to_wh.city if to_wh else "",
        "distance_km": route.distance_km,
        "ready_to_ship": route.ready_to_ship,
    }


# ── Vehicle types ─────────────────────────────────────────────────────────────

def get_all_vehicle_types(db: Session) -> List[VehicleType]:
    return db.query(VehicleType).all()


def get_vehicle_type_by_name(db: Session, name: str) -> Optional[VehicleType]:
    return db.query(VehicleType).filter(VehicleType.vehicle_type == name).first()


def vtype_to_dict(vt: VehicleType) -> Dict[str, Any]:
    return {
        "vehicle_type": vt.vehicle_type,
        "capacity_units": vt.capacity_units,
        "cost_per_km": vt.cost_per_km,
        "underload_penalty": vt.underload_penalty,
        "fixed_dispatch_cost": vt.fixed_dispatch_cost,
    }


# ── Per-warehouse fleet ──────────────────────────────────────────────────────

def get_fleet_for_warehouse(db: Session, warehouse_id: str) -> List[Dict[str, Any]]:
    """Return vehicles list in the format expected by vehicles_cfg['vehicles']."""
    rows = (
        db.query(WarehouseVehicle, VehicleType)
        .join(VehicleType, WarehouseVehicle.vehicle_type_id == VehicleType.id)
        .filter(WarehouseVehicle.warehouse_id == warehouse_id)
        .all()
    )
    return [
        {
            "vehicle_type": vt.vehicle_type,
            "capacity_units": vt.capacity_units,
            "cost_per_km": vt.cost_per_km,
            "available": wv.available,
            "underload_penalty": vt.underload_penalty,
            "fixed_dispatch_cost": vt.fixed_dispatch_cost,
        }
        for wv, vt in rows
    ]


def get_vehicles_cfg_for_warehouse(db: Session, warehouse_id: str) -> Dict[str, Any]:
    """Build a vehicles_cfg dict for a specific warehouse (compatible with optimizer)."""
    vehicles = get_fleet_for_warehouse(db, warehouse_id)
    settings = get_all_settings(db)
    return {
        "vehicles": vehicles,
        "wait_penalty_per_minute": settings.get("wait_penalty_per_minute", DEFAULT_WAIT_PENALTY_PER_MINUTE),
        "economy_threshold": settings.get("economy_threshold", DEFAULT_ECONOMY_THRESHOLD),
    }


# ── Incoming vehicles ─────────────────────────────────────────────────────────

def get_incoming_for_warehouse(db: Session, warehouse_id: str) -> List[Dict[str, Any]]:
    rows = (
        db.query(IncomingVehicle, VehicleType)
        .join(VehicleType, IncomingVehicle.vehicle_type_id == VehicleType.id)
        .filter(IncomingVehicle.warehouse_id == warehouse_id)
        .all()
    )
    return [
        {
            "horizon_idx": iv.horizon_idx,
            "vehicle_type": vt.vehicle_type,
            "count": iv.count,
        }
        for iv, vt in rows
    ]


# ── Settings ──────────────────────────────────────────────────────────────────

def get_all_settings(db: Session) -> Dict[str, Any]:
    rows = db.query(Setting).all()
    return {row.key: json.loads(row.value) for row in rows}


def get_setting(db: Session, key: str, default: Any = None) -> Any:
    row = db.query(Setting).filter(Setting.key == key).first()
    if row is None:
        return default
    return json.loads(row.value)


def set_setting(db: Session, key: str, value: Any) -> None:
    row = db.query(Setting).filter(Setting.key == key).first()
    encoded = json.dumps(value)
    if row is None:
        db.add(Setting(key=key, value=encoded))
    else:
        row.value = encoded


# ── Route distances (bulk) ────────────────────────────────────────────────────

def get_route_distances_list(db: Session) -> List[Dict[str, Any]]:
    """Return all routes in the legacy API format (with from_city/to_city)."""
    routes = get_all_routes(db)
    return [route_to_api_dict(r, db) for r in routes]


# ── Actuals from parquet ──────────────────────────────────────────────────────

def get_actuals_for_routes(
    route_ids: List[str],
    start_ts: datetime,
    end_ts: datetime,
    train_path: str | None = None,
) -> pd.DataFrame:
    """Read actual demand (target_2h) from the training parquet for given routes and date range.

    Returns DataFrame with columns: route_id, timestamp, target_2h.
    """
    path = train_path or str(DEFAULT_TRAIN_PATH)

    int_routes: list = []
    for r in route_ids:
        try:
            int_routes.append(int(r))
        except (ValueError, TypeError):
            pass

    df = None
    if int_routes:
        try:
            df = pd.read_parquet(
                path,
                columns=["route_id", "timestamp", "target_2h"],
                filters=[("route_id", "in", int_routes)],
            )
            if df.empty:
                df = None
        except Exception:
            df = None

    if df is None:
        try:
            df = pd.read_parquet(
                path,
                columns=["route_id", "timestamp", "target_2h"],
                filters=[("route_id", "in", route_ids)],
            )
        except Exception:
            logger.warning("Parquet predicate pushdown failed, falling back to full read")
            df = pd.read_parquet(path, columns=["route_id", "timestamp", "target_2h"])

    df["route_id"] = df["route_id"].astype(str)
    df = df[df["route_id"].isin(route_ids)]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df[df["timestamp"].notna()]
    df = df[(df["timestamp"] >= pd.Timestamp(start_ts)) & (df["timestamp"] <= pd.Timestamp(end_ts))]
    df = df.sort_values(["route_id", "timestamp"]).reset_index(drop=True)
    return df


# ── Dispatch results for retrospective analysis ──────────────────────────────

def get_dispatch_results_for_warehouse(
    db: Session,
    warehouse_id: str,
    start_ts: datetime,
    end_ts: datetime,
) -> List[DispatchResult]:
    """Return dispatch_results rows for a warehouse in a date range."""
    return (
        db.query(DispatchResult)
        .filter(
            DispatchResult.warehouse_id == warehouse_id,
            DispatchResult.timestamp >= start_ts,
            DispatchResult.timestamp <= end_ts,
            DispatchResult.granularity_2.isnot(None),
        )
        .order_by(DispatchResult.timestamp)
        .all()
    )


def get_dispatch_date_range(
    db: Session, warehouse_id: str
) -> Optional[Dict[str, Any]]:
    """Return min/max timestamp and count for dispatch_results with granularity_2 data."""
    row = (
        db.query(
            func.min(DispatchResult.timestamp),
            func.max(DispatchResult.timestamp),
            func.count(DispatchResult.id),
        )
        .filter(
            DispatchResult.warehouse_id == warehouse_id,
            DispatchResult.granularity_2.isnot(None),
        )
        .first()
    )
    if row is None or row[2] == 0:
        return None
    return {"min_date": row[0], "max_date": row[1], "count": row[2]}
