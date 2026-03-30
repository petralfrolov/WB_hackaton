from typing import List

from fastapi import APIRouter, Depends, HTTPException

from ml_prediction import predict_lazy
from schemas.warehouses import ForecastPoint, WarehouseInfo
from state import AppState, get_state

router = APIRouter(tags=["warehouses"])


@router.get("/warehouses", response_model=List[WarehouseInfo])
def list_warehouses(state: AppState = Depends(get_state)):
    return [WarehouseInfo(**w) for w in state.warehouses]


@router.get("/warehouses/{warehouse_id}", response_model=WarehouseInfo)
def get_warehouse(warehouse_id: str, state: AppState = Depends(get_state)):
    item = next((w for w in state.warehouses if w["id"] == warehouse_id), None)
    if item is None:
        raise HTTPException(status_code=404, detail="warehouse not found")
    return WarehouseInfo(**item)


@router.get("/warehouses/{warehouse_id}/forecast", response_model=List[ForecastPoint])
def get_warehouse_forecast(warehouse_id: str, timestamp: str, state: AppState = Depends(get_state)):
    warehouse = next((w for w in state.warehouses if w["id"] == warehouse_id), None)
    if warehouse is None:
        raise HTTPException(status_code=404, detail="warehouse not found")

    init_stock = float(state.vehicles_cfg.get("initial_stock_units", 0))
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

    points = [
        ForecastPoint(time=base_dt.strftime("%H:%M"), value=round(init_stock, 1), lower=round(init_stock * 0.9, 1), upper=round(init_stock * 1.1, 1)),
        ForecastPoint(time=(base_dt + timedelta(hours=2)).strftime("%H:%M"), value=round(totals[0], 1), lower=round(totals[0] * 0.9, 1), upper=round(totals[0] * 1.1, 1)),
        ForecastPoint(time=(base_dt + timedelta(hours=4)).strftime("%H:%M"), value=round(totals[1], 1), lower=round(totals[1] * 0.9, 1), upper=round(totals[1] * 1.1, 1)),
        ForecastPoint(time=(base_dt + timedelta(hours=6)).strftime("%H:%M"), value=round(totals[2], 1), lower=round(totals[2] * 0.9, 1), upper=round(totals[2] * 1.1, 1)),
    ]
    return points
