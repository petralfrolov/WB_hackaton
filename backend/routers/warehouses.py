from typing import List

from fastapi import APIRouter, Depends, HTTPException

from ml_prediction import predict_lazy
from schemas.warehouses import WarehouseInfo
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
