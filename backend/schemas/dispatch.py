from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel

from .optimize import PlanRow
from .vehicles import IncomingVehicle, Vehicle


class RouteMetrics(BaseModel):
    route_id: str
    fill_rate: float        # avg fraction of capacity used across dispatched horizons (0–1)
    cpo: float              # cost per shipped unit, ₽


class WarehouseMetrics(BaseModel):
    p_cover: float          # P(total demand ≤ total planned capacity), normal approx, across all horizons
    p_cover_by_horizon: List[float]   # per-horizon [A,B,C,D], A is always 1.0 (deterministic)
    fill_rate: float        # warehouse-level weighted fill rate (0–1)
    cpo: float              # warehouse-level cost per shipped unit, ₽
    route_metrics: List[RouteMetrics]
    horizon_labels: Optional[List[str]] = None  # dynamic labels based on granularity
    # Fleet utilization metrics
    fleet_utilization_ratio: Optional[float] = None   # required_capacity / available_capacity
    fleet_capacity_shortfall: Optional[float] = None  # required_capacity - available_capacity (units)
    required_capacity_units: Optional[float] = None   # total demand + conformal margin (units needed)
    available_capacity_units: Optional[float] = None  # total available fleet capacity (units)
    dispatched_capacity_units: Optional[float] = None  # what the MILP actually allocated (units)
    total_demand_units: Optional[float] = None         # raw demand without margin (units)
    total_conformal_margin: Optional[float] = None     # safety buffer from conformal prediction
    # Metric detail breakdowns (for clickable drill-down)
    p_cover_detail: Optional[List[Dict[str, Any]]] = None
    fill_rate_detail: Optional[List[Dict[str, Any]]] = None
    cpo_detail: Optional[List[Dict[str, Any]]] = None
    fleet_detail: Optional[List[Dict[str, Any]]] = None          # available fleet per vehicle type
    dispatched_detail: Optional[List[Dict[str, Any]]] = None     # dispatched per vehicle type
    utilization_detail: Optional[List[Dict[str, Any]]] = None    # demand vs capacity per horizon


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
    global_fleet: bool = True
    confidence_level: Optional[float] = None  # override state confidence_level
    granularity: Optional[Literal[0.5, 1.0, 2.0]] = None  # forecast granularity in hours


class DispatchResponse(BaseModel):
    warehouse_id: str
    office_from_id: str
    timestamp: datetime
    routes: List[RoutePlan]
    total_cost: float
    metrics: Optional[WarehouseMetrics] = None
    granularity: float = 2.0  # echoed back to frontend
    horizon_labels: Optional[List[str]] = None  # dynamic horizon labels
