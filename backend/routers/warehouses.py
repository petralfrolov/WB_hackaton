from typing import List

from fastapi import APIRouter, Depends, HTTPException

from conformal import conformal_interval, get_margin
from ml_prediction import predict_lazy
from schemas.warehouses import ForecastPoint, WarehouseInfo
from state import AppState, get_state

router = APIRouter(tags=["warehouses"])


def _route_meta_by_id(state: AppState) -> dict[str, dict]:
    return {str(route["id"]): route for route in state.route_distances}


def _warehouse_ready_to_ship(warehouse: dict, state: AppState) -> int:
    route_meta = _route_meta_by_id(state)
    return int(sum(int(route_meta.get(str(rid), {}).get("ready_to_ship", 0)) for rid in warehouse["route_ids"]))


@router.get("/warehouses", response_model=List[WarehouseInfo])
def list_warehouses(state: AppState = Depends(get_state)):
    items = []
    for warehouse in state.warehouses:
        payload = {**warehouse, "ready_to_ship": _warehouse_ready_to_ship(warehouse, state)}
        items.append(WarehouseInfo(**payload))
    return items


@router.get("/warehouses/{warehouse_id}", response_model=WarehouseInfo)
def get_warehouse(warehouse_id: str, state: AppState = Depends(get_state)):
    item = next((w for w in state.warehouses if w["id"] == warehouse_id), None)
    if item is None:
        raise HTTPException(status_code=404, detail="warehouse not found")
    return WarehouseInfo(**{**item, "ready_to_ship": _warehouse_ready_to_ship(item, state)})


@router.get("/warehouses/{warehouse_id}/forecast", response_model=List[ForecastPoint])
def get_warehouse_forecast(warehouse_id: str, timestamp: str, state: AppState = Depends(get_state)):
    warehouse = next((w for w in state.warehouses if w["id"] == warehouse_id), None)
    if warehouse is None:
        raise HTTPException(status_code=404, detail="warehouse not found")

    route_meta = _route_meta_by_id(state)
    init_stock = float(sum(float(route_meta.get(str(rid), {}).get("ready_to_ship", 0)) for rid in warehouse["route_ids"]))
    totals = [0.0, 0.0, 0.0]  # pred_0_2h, pred_2_4h, pred_4_6h

    valid_count = 0
    office_routes = state.office_routes_map.get(warehouse.get("office_from_id", ""), [])
    for rid in warehouse["route_ids"]:
        if rid not in state.office_map:
            continue
        preds = predict_lazy(
            train_path=state.train_path,
            models=state.models,
            route_id=rid,
            timestamp=timestamp,
            office_routes=office_routes,
        )
        totals[0] += preds["pred_0_2h"]
        totals[1] += preds["pred_2_4h"]
        totals[2] += preds["pred_4_6h"]
        valid_count += 1

    if valid_count == 0:
        raise HTTPException(status_code=422, detail="No routes found in training data for this warehouse")

    from datetime import datetime, timedelta
    try:
        base_dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        base_dt = datetime.now()

    margin_h1 = get_margin(state.ncs_scores, "__global__", "0-2h", state.confidence_level)
    margin_h2 = get_margin(state.ncs_scores, "__global__", "2-4h", state.confidence_level)
    margin_h3 = get_margin(state.ncs_scores, "__global__", "4-6h", state.confidence_level)
    # init_stock is the current observable stock — deterministic, no CI
    lo1, hi1 = conformal_interval(totals[0], margin_h1)
    lo2, hi2 = conformal_interval(totals[1], margin_h2)
    lo3, hi3 = conformal_interval(totals[2], margin_h3)

    points = [
        ForecastPoint(time=base_dt.strftime("%H:%M"), value=round(init_stock, 1), lower=round(init_stock, 1), upper=round(init_stock, 1)),
        ForecastPoint(time=(base_dt + timedelta(hours=2)).strftime("%H:%M"), value=round(totals[0], 1), lower=lo1, upper=hi1),
        ForecastPoint(time=(base_dt + timedelta(hours=4)).strftime("%H:%M"), value=round(totals[1], 1), lower=lo2, upper=hi2),
        ForecastPoint(time=(base_dt + timedelta(hours=6)).strftime("%H:%M"), value=round(totals[2], 1), lower=lo3, upper=hi3),
    ]
    return points
