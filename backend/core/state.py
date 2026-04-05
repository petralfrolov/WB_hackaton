"""Application-level state loaded once at startup and shared via FastAPI Depends()."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.conformal import NCS_DEFAULT_PATH, NCS_NORM_PATH, load_ncs
from ml.prediction import (
    DEFAULT_MODELS_DIR,
    DEFAULT_TRAIN_PATH,
    load_models,
)
from optimizer.horizons import load_route_office_map


@dataclass
class AppState:
    """Singleton application state loaded once at startup and shared via dependency injection.

    Attributes:
        models: Loaded LightGBM ensemble (list of model dicts from ``load_models``).
        train_path: Path to training parquet used for lazy per-request inference.
        office_map: Mapping ``route_id → office_from_id`` derived from parquet.
        office_routes_map: Inverse mapping ``office_from_id → [route_id, ...]``.
        vehicles_cfg: Parsed vehicles.json (vehicles list, penalties, thresholds).
        incoming_cfg: List of incoming-vehicle records from incoming_vehicles.json.
        warehouses: List of warehouse metadata dicts from warehouse_metadata.json.
        route_distances: List of route distance dicts from route_distances.json.
        last_dispatch_by_warehouse: Cache of most-recent dispatch result per warehouse.
        confidence_level: Conformal coverage level α ∈ (0, 1); default 0.9.
        route_correlation: Assumed inter-route demand correlation ρ ∈ [0, 1]; default 0.3.
        ncs_scores: Non-conformity score lookup dict (``NcsData``) from ``load_ncs``.
        ncs_normalized: True when NCS scores are relative ``|y-ŷ|/(ŷ+1)`` residuals.
    """
    models: Any = None
    train_path: Path = field(default_factory=lambda: Path(DEFAULT_TRAIN_PATH))
    office_map: Dict[str, str] = field(default_factory=dict)    # route_id → office_from_id
    office_routes_map: Dict[str, List[str]] = field(default_factory=dict)  # office_from_id → [route_ids]
    vehicles_cfg: Dict = field(default_factory=dict)
    incoming_cfg: List[Dict] = field(default_factory=list)
    warehouses: List[Dict] = field(default_factory=list)        # warehouse_metadata.json
    route_distances: List[Dict] = field(default_factory=list)   # route_distances.json
    last_dispatch_by_warehouse: Dict[str, Dict] = field(default_factory=dict)  # warehouse_id → last dispatch
    confidence_level: float = 0.9       # conformal coverage level (0-1)
    route_correlation: float = 0.3        # assumed inter-route demand correlation ρ ∈ [0,1]
    ncs_scores: Any = field(default_factory=lambda: load_ncs()[0])  # non-conformity scores
    ncs_normalized: bool = False          # True when scores are relative |y-ŷ|/(ŷ+1)


_state: Optional[AppState] = None


def get_state() -> AppState:
    """FastAPI dependency — returns the singleton AppState."""
    if _state is None:
        raise RuntimeError("AppState not initialised — call load_state() in lifespan first")
    return _state


def _load_json_configs(
    vehicles_path: Path,
    incoming_path: Path,
    warehouses_path: Path,
    route_distances_path: Path,
) -> tuple:
    """Read and parse the four JSON configuration files.

    Args:
        vehicles_path: Path to vehicles.json.
        incoming_path: Path to incoming_vehicles.json.
        warehouses_path: Path to warehouse_metadata.json.
        route_distances_path: Path to route_distances.json.

    Returns:
        Tuple (vehicles_cfg, incoming_cfg, warehouses, route_distances).
    """
    vehicles_cfg = json.loads(vehicles_path.read_text(encoding="utf-8"))
    incoming_raw = json.loads(incoming_path.read_text(encoding="utf-8"))
    incoming_cfg = incoming_raw.get("incoming", [])
    warehouses = json.loads(warehouses_path.read_text(encoding="utf-8")).get("warehouses", [])
    route_distances = json.loads(route_distances_path.read_text(encoding="utf-8")).get("route_distances", [])
    return vehicles_cfg, incoming_cfg, warehouses, route_distances


def _build_office_routes_map(office_map: Dict[str, str]) -> Dict[str, List[str]]:
    """Invert the route→office map into office→[routes] map.

    Args:
        office_map: Dict mapping ``route_id`` → ``office_from_id``.

    Returns:
        Dict mapping ``office_from_id`` → list of ``route_id`` strings.
    """
    result: Dict[str, List[str]] = {}
    for route_id, office_id in office_map.items():
        result.setdefault(office_id, []).append(route_id)
    return result


def load_state(
    train_path: Path = Path(DEFAULT_TRAIN_PATH),
    models_dir: Path = Path(DEFAULT_MODELS_DIR),
    vehicles_path: Path = Path("data/vehicles.json"),
    incoming_path: Path = Path("data/incoming_vehicles.json"),
    warehouses_path: Path = Path("data/warehouse_metadata.json"),
    route_distances_path: Path = Path("data/route_distances.json"),
) -> AppState:
    """Load all application state from disk and return a populated ``AppState``.

    Loads ML models, parquet office-map, JSON fleet/warehouse configs, and
    non-conformity scores. Sets the module-level singleton ``_state``.

    Args:
        train_path: Path to the training parquet used for office-map extraction
            and lazy inference.
        models_dir: Directory containing ``*.pkl`` LightGBM ensemble files.
        vehicles_path: Path to vehicles.json.
        incoming_path: Path to incoming_vehicles.json.
        warehouses_path: Path to warehouse_metadata.json.
        route_distances_path: Path to route_distances.json.

    Returns:
        The newly created and globally registered ``AppState``.
    """
    global _state

    models = load_models(models_dir)
    office_map = load_route_office_map(train_path)
    office_routes_map = _build_office_routes_map(office_map)

    vehicles_cfg, incoming_cfg, warehouses, route_distances = _load_json_configs(
        vehicles_path, incoming_path, warehouses_path, route_distances_path
    )

    ncs_scores, ncs_normalized = load_ncs(NCS_DEFAULT_PATH)

    _state = AppState(
        models=models,
        train_path=train_path,
        office_map=office_map,
        office_routes_map=office_routes_map,
        vehicles_cfg=vehicles_cfg,
        incoming_cfg=incoming_cfg,
        warehouses=warehouses,
        route_distances=route_distances,
        confidence_level=float(vehicles_cfg.get("confidence_level", 0.9)),
        ncs_scores=ncs_scores,
        ncs_normalized=ncs_normalized,
    )
    _state.last_dispatch = None
    return _state
