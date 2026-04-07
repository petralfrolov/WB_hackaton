from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class Settings(BaseModel):
    wait_penalty_per_minute: Optional[float] = None
    economy_threshold: Optional[float] = None
    confidence_level: Optional[float] = None  # conformal coverage, e.g. 0.90
    route_correlation: Optional[float] = None  # assumed inter-route demand correlation ρ ∈ [0,1]
    granularity: Optional[Literal[0.5, 1.0, 2.0]] = None  # forecast granularity in hours
