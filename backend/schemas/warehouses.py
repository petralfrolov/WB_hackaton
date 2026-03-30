from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel


class ForecastPoint(BaseModel):
    time: str
    value: float
    lower: float
    upper: float


class RouteDistance(BaseModel):
    id: str
    from_id: str
    to_id: str
    from_city: str
    to_city: str
    distance_km: float
    ready_to_ship: int = 0


WarehouseStatus = Literal["ok", "warning", "critical"]


class WarehouseInfo(BaseModel):
    id: str
    name: str
    city: str
    lat: float
    lng: float
    office_from_id: str
    route_ids: List[str]
    status: WarehouseStatus = "ok"
    ready_to_ship: int = 0
