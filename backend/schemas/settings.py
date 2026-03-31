from __future__ import annotations

from typing import Optional, Dict

from pydantic import BaseModel


class Settings(BaseModel):
    underload_penalty_per_unit: Optional[float] = None
    underload_penalty_per_unit_by_cat: Optional[Dict[str, float]] = None
    underload_penalty_compact: Optional[float] = None
    underload_penalty_mid: Optional[float] = None
    underload_penalty_large: Optional[float] = None
    wait_penalty_per_minute: Optional[float] = None
    initial_stock_units: Optional[float] = None
    route_distance_km: Optional[float] = None
    economy_threshold: Optional[float] = None
    max_wait_minutes: Optional[int] = None
