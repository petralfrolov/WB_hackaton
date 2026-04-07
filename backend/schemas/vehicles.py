from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Vehicle(BaseModel):
    vehicle_type: str
    capacity_units: float
    cost_per_km: float
    available: int
    underload_penalty: Optional[float] = None
    fixed_dispatch_cost: Optional[float] = None
    warehouse_id: Optional[str] = None  # when present, scoped to a specific warehouse


class VehicleUpdate(BaseModel):
    vehicle_type: Optional[str] = None
    capacity_units: Optional[float] = None
    cost_per_km: Optional[float] = None
    available: Optional[int] = None
    underload_penalty: Optional[float] = None
    fixed_dispatch_cost: Optional[float] = None
    warehouse_id: Optional[str] = None


class IncomingVehicle(BaseModel):
    horizon_idx: int = Field(..., ge=0)
    vehicle_type: str
    count: int = Field(..., ge=1)
    warehouse_id: Optional[str] = None


class IncomingVehicleList(BaseModel):
    incoming: List[IncomingVehicle] = []
