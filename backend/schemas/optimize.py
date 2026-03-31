from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class OptimizeRequest(BaseModel):
    route_id: str
    timestamp: datetime
    wait_penalty_per_minute: Optional[float] = None


class PlanRow(BaseModel):
    office_from_id: Optional[str] = None
    route_id: str
    timestamp: datetime
    horizon: str
    vehicle_type: str
    vehicles_count: int
    demand_new: float
    demand_carried_over: float
    total_available: float
    actually_shipped: float
    leftover_stock: float
    empty_capacity_units: float
    cost_fixed: float
    cost_underload: float
    cost_wait: float
    cost_total: float


class OptimizeResponse(BaseModel):
    plan: List[PlanRow]
    coverage_min: float
