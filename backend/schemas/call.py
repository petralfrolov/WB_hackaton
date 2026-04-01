from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class CallRequest(BaseModel):
    route_id: str
    timestamp: datetime
    warehouse_id: Optional[str] = None


class CallVehicle(BaseModel):
    vehicle_type: str
    vehicles_count: int
    category: Optional[str] = None
    capacity_units: float
    cost_per_km: float
    empty_capacity_units: float
    cost_fixed: float
    cost_underload: float


class CallPayload(BaseModel):
    route_id: str
    office_from_id: Optional[str] = None
    dispatch_time: datetime
    horizon: str
    vehicles: List[CallVehicle]
    costs: dict
    demand: dict


class CallResponse(BaseModel):
    request: CallPayload
