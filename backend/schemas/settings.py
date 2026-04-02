from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class Settings(BaseModel):
    wait_penalty_per_minute: Optional[float] = None
    economy_threshold: Optional[float] = None
    max_wait_minutes: Optional[int] = None
    confidence_level: Optional[float] = None  # conformal coverage, e.g. 0.90
