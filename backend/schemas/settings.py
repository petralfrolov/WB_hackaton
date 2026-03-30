from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class Settings(BaseModel):
    underload_penalty_per_unit: Optional[float] = None
    wait_penalty_per_minute: Optional[float] = None
    initial_stock_units: Optional[float] = None
    route_distance_km: Optional[float] = None
    economy_threshold: Optional[float] = None
    max_wait_minutes: Optional[int] = None
