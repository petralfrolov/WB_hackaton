"""Minimal FastAPI app wrapping horizon optimizer.

Run for dev:
    uvicorn api:app --reload --port 8000

Dependencies (install once):
    pip install fastapi uvicorn
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml_prediction import (
    DEFAULT_MODELS_DIR,
    DEFAULT_TRAIN_PATH,
    load_models,
    prepare_feature_matrix,
    predict_for_route_timestamp,
)
from optimizer_horizons import build_plan, load_route_office_map


# ------------ Pydantic models ------------

class OptimizeRequest(BaseModel):
    route_id: str
    timestamp: datetime
    initial_stock_units: Optional[float] = None
    route_distance_km: Optional[float] = None
    wait_penalty_per_minute: Optional[float] = None
    underload_penalty_per_unit: Optional[float] = None


class PlanRow(BaseModel):
    office_from_id: Optional[str]
    route_id: str
    timestamp: datetime
    horizon: str
    vehicle_type: str
    vehicles_count: int
    demand: float
    covered: float
    cost_fixed: float
    cost_underload: float
    cost_wait: float
    cost_total: float


class OptimizeResponse(BaseModel):
    plan: List[PlanRow]
    coverage_min: float


class Vehicle(BaseModel):
    vehicle_type: str
    capacity_units: float
    cost_per_km: float
    available: int


class Settings(BaseModel):
    underload_penalty_per_unit: Optional[float] = None
    wait_penalty_per_minute: Optional[float] = None
    initial_stock_units: Optional[float] = None
    route_distance_km: Optional[float] = None


# ------------ App init ------------

app = FastAPI(title="WB Transport Optimizer", version="0.1")


@app.on_event("startup")
def _load_artifacts():
    global MODELS, X_ALL, FEATURE_COLS, OFFICE_MAP, VEHICLES_CFG
    train_path = Path(DEFAULT_TRAIN_PATH)
    models_dir = Path(DEFAULT_MODELS_DIR)
    vehicles_path = Path("vehicles.json")

    MODELS = load_models(models_dir)
    X_ALL, FEATURE_COLS = prepare_feature_matrix(train_path)
    OFFICE_MAP = load_route_office_map(train_path)
    VEHICLES_CFG = json.loads(vehicles_path.read_text())


# ------------ Helpers ------------

def apply_overrides(cfg: dict, req: OptimizeRequest) -> dict:
    out = json.loads(json.dumps(cfg))  # deep copy via json
    if req.initial_stock_units is not None:
        out["initial_stock_units"] = float(req.initial_stock_units)
    if req.route_distance_km is not None:
        out["route_distance_km"] = float(req.route_distance_km)
    if req.wait_penalty_per_minute is not None:
        out["wait_penalty_per_minute"] = float(req.wait_penalty_per_minute)
    if req.underload_penalty_per_unit is not None:
        out["underload_penalty_per_unit"] = float(req.underload_penalty_per_unit)
    return out


# ------------ Endpoints ------------

@app.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest):
    route_id = str(req.route_id)
    ts_str = req.timestamp.strftime("%Y-%m-%d %H:%M:%S")

    if route_id not in OFFICE_MAP:
        raise HTTPException(status_code=404, detail="route_id not found in train")

    preds = predict_for_route_timestamp(
        X_all=X_ALL,
        feature_cols=FEATURE_COLS,
        models=MODELS,
        route_id=route_id,
        timestamp=ts_str,
    )

    cfg = apply_overrides(VEHICLES_CFG, req)
    plan_df = build_plan(route_id, ts_str, preds, cfg)
    office_id = OFFICE_MAP.get(route_id)
    if office_id:
        plan_df.insert(0, "office_from_id", office_id)
    plan_df["timestamp"] = pd.to_datetime(plan_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    coverage_min = float((plan_df["covered"] - plan_df["demand"]).min()) if not plan_df.empty else 0.0

    return OptimizeResponse(
        plan=[PlanRow(**r) for r in plan_df.to_dict("records")],
        coverage_min=coverage_min,
    )


@app.get("/config")
def get_config():
    return VEHICLES_CFG


@app.post("/config")
def set_config(payload: dict):
    global VEHICLES_CFG
    VEHICLES_CFG = payload
    Path("vehicles.json").write_text(json.dumps(VEHICLES_CFG, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "ok"}


# ---------- Vehicles CRUD ----------

@app.get("/vehicles", response_model=List[Vehicle])
def list_vehicles():
    vehicles = VEHICLES_CFG.get("vehicles", []) if isinstance(VEHICLES_CFG, dict) else VEHICLES_CFG
    return [Vehicle(**v) for v in vehicles]


@app.post("/vehicles", response_model=Vehicle)
def add_vehicle(v: Vehicle):
    vehicles = VEHICLES_CFG.get("vehicles", []) if isinstance(VEHICLES_CFG, dict) else VEHICLES_CFG
    if any(x.get("vehicle_type") == v.vehicle_type for x in vehicles):
        raise HTTPException(status_code=400, detail="vehicle_type already exists")
    vehicles.append(v.dict())
    VEHICLES_CFG["vehicles"] = vehicles
    Path("vehicles.json").write_text(json.dumps(VEHICLES_CFG, ensure_ascii=False, indent=2), encoding="utf-8")
    return v


@app.patch("/vehicles/{vehicle_type}", response_model=Vehicle)
def update_vehicle(vehicle_type: str, v: Vehicle):
    vehicles = VEHICLES_CFG.get("vehicles", []) if isinstance(VEHICLES_CFG, dict) else VEHICLES_CFG
    found = False
    for i, item in enumerate(vehicles):
        if item.get("vehicle_type") == vehicle_type:
            vehicles[i] = v.dict()
            found = True
            break
    if not found:
        raise HTTPException(status_code=404, detail="vehicle_type not found")
    VEHICLES_CFG["vehicles"] = vehicles
    Path("vehicles.json").write_text(json.dumps(VEHICLES_CFG, ensure_ascii=False, indent=2), encoding="utf-8")
    return v


@app.delete("/vehicles/{vehicle_type}")
def delete_vehicle(vehicle_type: str):
    vehicles = VEHICLES_CFG.get("vehicles", []) if isinstance(VEHICLES_CFG, dict) else VEHICLES_CFG
    new_list = [v for v in vehicles if v.get("vehicle_type") != vehicle_type]
    if len(new_list) == len(vehicles):
        raise HTTPException(status_code=404, detail="vehicle_type not found")
    VEHICLES_CFG["vehicles"] = new_list
    Path("vehicles.json").write_text(json.dumps(VEHICLES_CFG, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "ok"}


# ---------- Settings PATCH ----------

@app.patch("/settings")
def update_settings(s: Settings):
    for key, val in s.dict(exclude_none=True).items():
        VEHICLES_CFG[key] = val
    Path("vehicles.json").write_text(json.dumps(VEHICLES_CFG, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "ok", "settings": {k: VEHICLES_CFG.get(k) for k in s.dict(exclude_none=True)}}


@app.get("/health")
def health():
    return {"status": "ok"}
