"""Retrospective business metrics router.

Compares historical dispatch forecasts (from dispatch_results.granularity_2)
against actual demand (from the training parquet) to produce accuracy and
business KPIs.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db.database import get_db
from db.queries import (
    get_actuals_for_routes,
    get_dispatch_date_range,
    get_dispatch_results_for_warehouse,
    get_route_ids_for_warehouse,
    get_warehouse_by_id,
)
from schemas.metrics import (
    AggregateMetrics,
    AvailableDates,
    ForecastVsActual,
    MetricsRequest,
    MetricsResponse,
    RouteSummary,
    TimeSeriesMetric,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/metrics", tags=["metrics"])

# Mapping: horizon label → offset in hours used to align actual timestamp.
# Horizon "B: +2h" predicts demand in [now, now+2h], stored in target_2h at
# timestamp now+2h (rolling 2h sum).  We match by shifting the dispatch
# timestamp forward by the horizon offset.
_HORIZON_OFFSET_HOURS: Dict[str, int] = {
    "A: now": 0,
    "B: +2h": 2,
    "C: +4h": 4,
    "D: +6h": 6,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_dispatch_json(raw: str) -> Dict[str, Any] | None:
    """Safely parse the granularity_2 JSON column."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def _extract_plan_rows_for_horizon(
    dispatch_json: Dict[str, Any],
    horizon: str,
) -> List[Dict[str, Any]]:
    """Extract PlanRow dicts for a specific horizon from a DispatchResponse JSON."""
    rows: List[Dict[str, Any]] = []
    for route_plan in dispatch_json.get("routes", []):
        for pr in route_plan.get("plan", []):
            if pr.get("horizon") == horizon:
                rows.append(pr)
    return rows


def _aggregate_by_route_horizon(
    plan_rows: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Aggregate plan rows by route_id (summing across vehicle types).

    Returns {route_id: {demand_new, demand_lower, demand_upper,
                        actually_shipped, cost_total, empty_capacity}}.
    """
    agg: Dict[str, Dict[str, float]] = {}
    for pr in plan_rows:
        rid = pr["route_id"]
        if rid not in agg:
            agg[rid] = {
                "demand_new": pr.get("demand_new", 0.0),
                "demand_lower": pr.get("demand_lower", 0.0),
                "demand_upper": pr.get("demand_upper", 0.0),
                "actually_shipped": 0.0,
                "cost_total": 0.0,
                "empty_capacity": 0.0,
            }
        # demand_new / lower / upper are the same across vehicle types for one
        # route×horizon, but shipped / cost are per vehicle type → sum them.
        agg[rid]["actually_shipped"] += pr.get("actually_shipped", 0.0)
        agg[rid]["cost_total"] += pr.get("cost_total", 0.0)
        agg[rid]["empty_capacity"] += pr.get("empty_capacity_units", 0.0)
    return agg


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/available-dates/{warehouse_id}", response_model=AvailableDates)
def available_dates(warehouse_id: str, db: Session = Depends(get_db)):
    wh = get_warehouse_by_id(db, warehouse_id)
    if not wh:
        raise HTTPException(status_code=404, detail=f"Warehouse {warehouse_id} not found")
    info = get_dispatch_date_range(db, warehouse_id)
    if info is None:
        return AvailableDates(count=0)
    return AvailableDates(
        min_date=info["min_date"],
        max_date=info["max_date"],
        count=info["count"],
    )


@router.post("/retrospective", response_model=MetricsResponse)
def retrospective_metrics(req: MetricsRequest, db: Session = Depends(get_db)):
    # ── Validate ──────────────────────────────────────────────────────────
    if req.horizon not in _HORIZON_OFFSET_HOURS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid horizon '{req.horizon}'. Must be one of {list(_HORIZON_OFFSET_HOURS.keys())}",
        )

    wh = get_warehouse_by_id(db, req.warehouse_id)
    if not wh:
        raise HTTPException(status_code=404, detail=f"Warehouse {req.warehouse_id} not found")

    route_ids = get_route_ids_for_warehouse(db, req.warehouse_id)
    if not route_ids:
        raise HTTPException(status_code=404, detail="No routes found for this warehouse")

    # ── Load dispatch results ─────────────────────────────────────────────
    results = get_dispatch_results_for_warehouse(db, req.warehouse_id, req.date_from, req.date_to)
    if not results:
        raise HTTPException(status_code=404, detail="No dispatch results found for the given date range")

    offset_hours = _HORIZON_OFFSET_HOURS[req.horizon]

    # ── Build forecast records keyed by (route_id, dispatch_timestamp) ────
    # forecast_records: list of {route_id, dispatch_ts, predicted, ci_lower, ci_upper,
    #                            actually_shipped, cost_total, capacity_sent}
    forecast_records: List[Dict[str, Any]] = []

    for dr in results:
        djson = _parse_dispatch_json(dr.granularity_2)
        if djson is None:
            continue

        plan_rows = _extract_plan_rows_for_horizon(djson, req.horizon)
        agg = _aggregate_by_route_horizon(plan_rows)

        for rid, vals in agg.items():
            forecast_records.append({
                "route_id": rid,
                "dispatch_ts": dr.timestamp,
                "predicted": vals["demand_new"],
                "ci_lower": vals["demand_lower"],
                "ci_upper": vals["demand_upper"],
                "actually_shipped": vals["actually_shipped"],
                "cost_total": vals["cost_total"],
                "capacity_sent": vals["actually_shipped"] + vals["empty_capacity"],
            })

    if not forecast_records:
        raise HTTPException(status_code=404, detail="No plan rows found for the selected horizon")

    # ── Load actuals from parquet ─────────────────────────────────────────
    import pandas as pd
    from datetime import timedelta

    actuals_df = get_actuals_for_routes(route_ids, req.date_from, req.date_to + timedelta(hours=8))
    # Build lookup: (route_id, timestamp) → target_2h
    actuals_lookup: Dict[tuple, float] = {}
    for _, row in actuals_df.iterrows():
        key = (str(row["route_id"]), row["timestamp"])
        actuals_lookup[key] = float(row["target_2h"])

    # ── Join forecast vs actual ───────────────────────────────────────────
    forecast_vs_actual: List[ForecastVsActual] = []
    # Per-timestamp aggregators for time series
    ts_agg: Dict[str, Dict[str, float]] = {}  # dispatch_ts → {sum_pred, sum_actual, sum_abs_err, ...}
    # Per-route aggregators
    route_agg: Dict[str, Dict[str, float]] = {}

    for rec in forecast_records:
        # The actual at the moment matching this horizon
        actual_ts = rec["dispatch_ts"] + timedelta(hours=offset_hours)
        actual_key = (rec["route_id"], pd.Timestamp(actual_ts))
        actual_val = actuals_lookup.get(actual_key)
        if actual_val is None:
            # Try rounding to nearest 30-min slot
            rounded = actual_ts.replace(second=0, microsecond=0)
            minute = rounded.minute
            rounded = rounded.replace(minute=(minute // 30) * 30)
            actual_key = (rec["route_id"], pd.Timestamp(rounded))
            actual_val = actuals_lookup.get(actual_key)

        if actual_val is None:
            continue

        predicted = rec["predicted"]
        fva = ForecastVsActual(
            route_id=rec["route_id"],
            timestamp=rec["dispatch_ts"],
            horizon=req.horizon,
            predicted=round(predicted, 2),
            actual=round(actual_val, 2),
            ci_lower=round(rec["ci_lower"], 2),
            ci_upper=round(rec["ci_upper"], 2),
        )
        forecast_vs_actual.append(fva)

        # Time series aggregation
        ts_key = str(rec["dispatch_ts"])
        if ts_key not in ts_agg:
            ts_agg[ts_key] = {
                "sum_pred": 0, "sum_actual": 0, "sum_abs_err": 0,
                "sum_shipped": 0, "sum_cost": 0, "sum_capacity": 0,
            }
        ts_agg[ts_key]["sum_pred"] += predicted
        ts_agg[ts_key]["sum_actual"] += actual_val
        ts_agg[ts_key]["sum_abs_err"] += abs(actual_val - predicted)
        ts_agg[ts_key]["sum_shipped"] += rec["actually_shipped"]
        ts_agg[ts_key]["sum_cost"] += rec["cost_total"]
        ts_agg[ts_key]["sum_capacity"] += rec["capacity_sent"]

        # Route aggregation
        rid = rec["route_id"]
        if rid not in route_agg:
            route_agg[rid] = {
                "sum_pred": 0, "sum_actual": 0, "sum_abs_err": 0, "sum_signed_err": 0,
                "sum_shipped": 0, "sum_cost": 0, "sum_capacity": 0, "count": 0,
            }
        route_agg[rid]["sum_pred"] += predicted
        route_agg[rid]["sum_actual"] += actual_val
        route_agg[rid]["sum_abs_err"] += abs(actual_val - predicted)
        route_agg[rid]["sum_signed_err"] += (predicted - actual_val)
        route_agg[rid]["sum_shipped"] += rec["actually_shipped"]
        route_agg[rid]["sum_cost"] += rec["cost_total"]
        route_agg[rid]["sum_capacity"] += rec["capacity_sent"]
        route_agg[rid]["count"] += 1

    if not forecast_vs_actual:
        raise HTTPException(
            status_code=404,
            detail="No matching actuals found in parquet for the selected date range and horizon",
        )

    # ── Compute aggregate metrics ─────────────────────────────────────────
    total_pred = sum(fva.predicted for fva in forecast_vs_actual)
    total_actual = sum(fva.actual for fva in forecast_vs_actual)
    total_abs_err = sum(abs(fva.actual - fva.predicted) for fva in forecast_vs_actual)
    total_signed_err = sum(fva.predicted - fva.actual for fva in forecast_vs_actual)
    in_ci = sum(1 for fva in forecast_vs_actual if fva.ci_lower <= fva.actual <= fva.ci_upper)

    total_shipped_all = sum(r["actually_shipped"] for r in forecast_records if r["route_id"] in route_agg)
    total_cost_all = sum(r["cost_total"] for r in forecast_records if r["route_id"] in route_agg)
    total_capacity_all = sum(r["capacity_sent"] for r in forecast_records if r["route_id"] in route_agg)

    n = len(forecast_vs_actual)
    wape = round(total_abs_err / total_actual, 4) if total_actual > 0 else 0.0
    bias = round(total_signed_err / n, 2) if n > 0 else 0.0
    avg_cpo = round(total_cost_all / total_shipped_all, 2) if total_shipped_all > 0 else 0.0
    avg_fill = round(total_shipped_all / total_capacity_all, 4) if total_capacity_all > 0 else 0.0
    realized_cov = round(in_ci / n, 4) if n > 0 else 0.0

    aggregate = AggregateMetrics(
        avg_cpo=avg_cpo,
        avg_fill_rate=avg_fill,
        wape=wape,
        bias=bias,
        realized_coverage=realized_cov,
    )

    # ── Time series metrics ───────────────────────────────────────────────
    time_series: List[TimeSeriesMetric] = []
    for ts_str in sorted(ts_agg.keys()):
        a = ts_agg[ts_str]
        ts_wape = round(a["sum_abs_err"] / a["sum_actual"], 4) if a["sum_actual"] > 0 else 0.0
        ts_cpo = round(a["sum_cost"] / a["sum_shipped"], 2) if a["sum_shipped"] > 0 else 0.0
        ts_fill = round(a["sum_shipped"] / a["sum_capacity"], 4) if a["sum_capacity"] > 0 else 0.0
        time_series.append(TimeSeriesMetric(
            timestamp=ts_str,
            cpo=ts_cpo,
            fill_rate=ts_fill,
            wape=ts_wape,
        ))

    # ── Route summaries ───────────────────────────────────────────────────
    route_summary: List[RouteSummary] = []
    for rid, a in sorted(route_agg.items()):
        cnt = a["count"]
        r_wape = round(a["sum_abs_err"] / a["sum_actual"], 4) if a["sum_actual"] > 0 else 0.0
        r_bias = round(a["sum_signed_err"] / cnt, 2) if cnt > 0 else 0.0
        r_cpo = round(a["sum_cost"] / a["sum_shipped"], 2) if a["sum_shipped"] > 0 else 0.0
        r_fill = round(a["sum_shipped"] / a["sum_capacity"], 4) if a["sum_capacity"] > 0 else 0.0
        route_summary.append(RouteSummary(
            route_id=rid,
            avg_predicted=round(a["sum_pred"] / cnt, 2) if cnt > 0 else 0.0,
            avg_actual=round(a["sum_actual"] / cnt, 2) if cnt > 0 else 0.0,
            wape=r_wape,
            bias=r_bias,
            fill_rate=r_fill,
            cpo=r_cpo,
        ))

    return MetricsResponse(
        forecast_vs_actual=forecast_vs_actual,
        aggregate=aggregate,
        time_series=time_series,
        route_summary=route_summary,
    )
