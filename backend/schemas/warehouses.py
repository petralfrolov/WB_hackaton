from __future__ import annotations

from typing import List, Optional

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


class WarehouseInfo(BaseModel):
    id: str
    name: str
    city: str
    lat: Optional[float] = None
    lng: Optional[float] = None
    office_from_id: Optional[str] = None
    route_ids: List[str] = []
    ready_to_ship: int = 0
