"""Pydantic schemas for the retrospective business metrics endpoint."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class MetricsRequest(BaseModel):
    warehouse_id: str
    date_from: datetime
    date_to: datetime
    horizon: str = "B: +2h"  # one of "B: +2h", "C: +4h", "D: +6h"


class ForecastVsActual(BaseModel):
    route_id: str
    timestamp: datetime
    horizon: str
    predicted: float
    actual: float
    ci_lower: float
    ci_upper: float


class AggregateMetrics(BaseModel):
    avg_cpo: float
    avg_fill_rate: float
    wape: float          # weighted absolute percentage error
    bias: float          # mean signed error (predicted - actual)
    realized_coverage: float  # fraction of actuals within CI


class TimeSeriesMetric(BaseModel):
    timestamp: datetime
    cpo: float
    fill_rate: float
    wape: float


class RouteSummary(BaseModel):
    route_id: str
    avg_predicted: float
    avg_actual: float
    wape: float
    bias: float
    fill_rate: float
    cpo: float


class AvailableDates(BaseModel):
    min_date: Optional[datetime] = None
    max_date: Optional[datetime] = None
    count: int = 0


class MetricsResponse(BaseModel):
    forecast_vs_actual: List[ForecastVsActual]
    aggregate: AggregateMetrics
    time_series: List[TimeSeriesMetric]
    route_summary: List[RouteSummary]
