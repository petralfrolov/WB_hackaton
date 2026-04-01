"""Application-level state loaded once at startup and shared via FastAPI Depends()."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ml_prediction import (
    DEFAULT_MODELS_DIR,
    DEFAULT_TRAIN_PATH,
    load_models,
)
from optimizer_horizons import load_route_office_map


@dataclass
class AppState:
    models: Any = None
    train_path: Path = field(default_factory=lambda: Path(DEFAULT_TRAIN_PATH))
    office_map: Dict[str, str] = field(default_factory=dict)    # route_id → office_from_id
    office_routes_map: Dict[str, List[str]] = field(default_factory=dict)  # office_from_id → [route_ids]
    vehicles_cfg: Dict = field(default_factory=dict)
    incoming_cfg: List[Dict] = field(default_factory=list)
    warehouses: List[Dict] = field(default_factory=list)        # warehouse_metadata.json
    route_distances: List[Dict] = field(default_factory=list)   # route_distances.json
    last_dispatch_by_warehouse: Dict[str, Dict] = field(default_factory=dict)  # warehouse_id → last dispatch


_state: Optional[AppState] = None


def get_state() -> AppState:
    """FastAPI dependency — returns the singleton AppState."""
    if _state is None:
        raise RuntimeError("AppState not initialised — call load_state() in lifespan first")
    return _state


def load_state(
    train_path: Path = Path(DEFAULT_TRAIN_PATH),
    models_dir: Path = Path(DEFAULT_MODELS_DIR),
    vehicles_path: Path = Path("vehicles.json"),
    incoming_path: Path = Path("incoming_vehicles.json"),
    warehouses_path: Path = Path("warehouse_metadata.json"),
    route_distances_path: Path = Path("route_distances.json"),
) -> AppState:
    global _state

    models = load_models(models_dir)
    office_map = load_route_office_map(train_path)

    # Reverse map: office_from_id → list of route_ids
    office_routes_map: Dict[str, List[str]] = {}
    for route_id, office_id in office_map.items():
        office_routes_map.setdefault(office_id, []).append(route_id)

    vehicles_cfg = json.loads(vehicles_path.read_text(encoding="utf-8"))
    incoming_raw = json.loads(incoming_path.read_text(encoding="utf-8"))
    incoming_cfg = incoming_raw.get("incoming", [])
    warehouses = json.loads(warehouses_path.read_text(encoding="utf-8")).get("warehouses", [])
    route_distances = json.loads(route_distances_path.read_text(encoding="utf-8")).get("route_distances", [])

    _state = AppState(
        models=models,
        train_path=train_path,
        office_map=office_map,
        office_routes_map=office_routes_map,
        vehicles_cfg=vehicles_cfg,
        incoming_cfg=incoming_cfg,
        warehouses=warehouses,
        route_distances=route_distances,
    )
    # cache for last dispatch result
    _state.last_dispatch = None
    return _state
