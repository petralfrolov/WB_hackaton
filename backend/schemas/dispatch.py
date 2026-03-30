from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel

from .optimize import PlanRow
from .vehicles import IncomingVehicle, Vehicle


class RoutePlan(BaseModel):
    route_id: str
    plan: List[PlanRow]
    coverage_min: float


class DispatchRequest(BaseModel):
    warehouse_id: str
    timestamp: datetime
    vehicles_override: Optional[List[Vehicle]] = None
    incoming_vehicles: Optional[List[IncomingVehicle]] = None
    wait_penalty_per_minute: Optional[float] = None
    underload_penalty_per_unit: Optional[float] = None
    global_fleet: bool = False


class DispatchResponse(BaseModel):
    warehouse_id: str
    office_from_id: str
    timestamp: datetime
    routes: List[RoutePlan]
    total_cost: float
