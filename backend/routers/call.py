"""Transport call request builder router.

Handles POST /call: formats a structured transport call payload from the most
recent cached dispatch plan for a given route and warehouse.
"""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from schemas.call import CallRequest, CallResponse, CallVehicle, CallPayload
from core.state import AppState, get_state

router = APIRouter(tags=["call"])


@router.post("/call", response_model=CallResponse)
async def call_transport(req: CallRequest, state: AppState = Depends(get_state)):
    """Формирует JSON вызова транспорта, используя уже готовый план из DispatchResponse.

    Ожидается, что фронт передаёт маршрут, который присутствует в последнем dispatch для склада.
    """
    route_id = str(req.route_id)
    ts_str = req.timestamp.strftime("%Y-%m-%d %H:%M:%S")

    # Ищем план: сначала по warehouse_id, затем общий last_dispatch
    last_dispatch = None
    if req.warehouse_id:
        last_dispatch = state.last_dispatch_by_warehouse.get(str(req.warehouse_id))
    if not last_dispatch:
        last_dispatch = getattr(state, "last_dispatch", None)
    if not last_dispatch:
        raise HTTPException(status_code=409, detail="Нет готового плана. Сначала нажмите «Обновить прогноз».")

    routes = {r["route_id"]: r for r in last_dispatch.get("routes", [])}
    if route_id not in routes:
        raise HTTPException(status_code=404, detail="Маршрут не найден в последнем плане")

    plan = routes[route_id]["plan"]
    a_rows = [r for r in plan if r["horizon"] == "A: now" and r["vehicles_count"] > 0]
    if not a_rows:
        raise HTTPException(status_code=422, detail="В горизонте A нет отправок для этого маршрута")

    vehicles_cfg = state.vehicles_cfg.get("vehicles", [])
    v_lookup = {v["vehicle_type"]: v for v in vehicles_cfg}

    vehicles = []
    for row in a_rows:
        v_cfg = v_lookup.get(row["vehicle_type"], {})
        vehicles.append(CallVehicle(
            vehicle_type=row["vehicle_type"],
            vehicles_count=int(row["vehicles_count"]),
            category=row.get("vehicle_category") or v_cfg.get("category"),
            capacity_units=float(v_cfg.get("capacity_units", row.get("capacity_units", 0))),
            cost_per_km=float(v_cfg.get("cost_per_km", row.get("cost_per_km", 0))),
            empty_capacity_units=row["empty_capacity_units"],
            cost_fixed=row["cost_fixed"],
            cost_underload=row["cost_underload"],
        ))

    costs = {
        "fixed": sum(r["cost_fixed"] for r in a_rows),
        "underload": sum(r["cost_underload"] for r in a_rows),
        "wait": sum(r["cost_wait"] for r in a_rows),
        "total": sum(r["cost_total"] for r in a_rows),
    }

    # ready_to_ship берём как сумма demand_new + demand_carried_over первой строки A
    ready_sum = 0.0
    if a_rows:
        ready_sum = float(a_rows[0].get("demand_new", 0) + a_rows[0].get("demand_carried_over", 0))

    payload = CallPayload(
        route_id=route_id,
        office_from_id=routes[route_id].get("office_from_id"),
        dispatch_time=req.timestamp,
        horizon="A: now",
        vehicles=vehicles,
        costs=costs,
        demand={
          "ready_to_ship": ready_sum,
        },
    )
    return CallResponse(request=payload)
