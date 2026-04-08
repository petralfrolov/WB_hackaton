"""Application-level state loaded once at startup and shared via FastAPI Depends()."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import (
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_DISPATCH_CONCURRENCY,
    DEFAULT_GRANULARITY_HOURS,
    DEFAULT_MODELS_DIR,
    DEFAULT_ROUTE_CORRELATION,
    DEFAULT_TRAIN_PATH,
)
from core.conformal import NCS_DEFAULT_PATH, load_ncs, load_ncs_allsteps
from ml.prediction import (
    load_models,
)
from optimizer.horizons import load_route_office_map
from db.database import SessionLocal


@dataclass
class AppState:
    """Singleton application state loaded once at startup and shared via dependency injection."""
    models: Any = None
    train_path: Path = field(default_factory=lambda: DEFAULT_TRAIN_PATH)
    office_map: Dict[str, str] = field(default_factory=dict)    # route_id → office_from_id
    office_routes_map: Dict[str, List[str]] = field(default_factory=dict)  # office_from_id → [route_ids]
    last_dispatch_by_warehouse: Dict[str, Dict] = field(default_factory=dict)
    last_dispatch: Optional[Dict] = None
    _dispatch_write_lock: threading.Lock = field(default_factory=threading.Lock)
    _warehouse_locks: Dict[str, threading.Lock] = field(default_factory=dict)
    _dispatch_semaphore: threading.Semaphore = field(
        default_factory=lambda: threading.Semaphore(DEFAULT_DISPATCH_CONCURRENCY)
    )
    _active_dispatches: int = 0
    _active_dispatches_lock: threading.Lock = field(default_factory=threading.Lock)
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
    route_correlation: float = DEFAULT_ROUTE_CORRELATION
    ncs_scores: Any = field(default_factory=lambda: load_ncs()[0])
    ncs_normalized: bool = False
    ncs_allsteps: Any = field(default_factory=dict)
    granularity: float = DEFAULT_GRANULARITY_HOURS

    @property
    def dispatching(self) -> bool:
        """True when at least one dispatch is in progress."""
        return self._active_dispatches > 0

    def inc_dispatches(self) -> None:
        with self._active_dispatches_lock:
            self._active_dispatches += 1

    def dec_dispatches(self) -> None:
        with self._active_dispatches_lock:
            self._active_dispatches = max(0, self._active_dispatches - 1)

    def get_warehouse_lock(self, warehouse_id: str) -> threading.Lock:
        """Return (creating if needed) the per-warehouse dispatch lock."""
        if warehouse_id not in self._warehouse_locks:
            with self._dispatch_write_lock:
                if warehouse_id not in self._warehouse_locks:
                    self._warehouse_locks[warehouse_id] = threading.Lock()
        return self._warehouse_locks[warehouse_id]


_state: Optional[AppState] = None


def get_state() -> AppState:
    """FastAPI dependency — returns the singleton AppState."""
    if _state is None:
        raise RuntimeError("AppState not initialised — call load_state() in lifespan first")
    return _state


def _build_office_routes_map(office_map: Dict[str, str]) -> Dict[str, List[str]]:
    """Invert the route→office map into office→[routes] map."""
    result: Dict[str, List[str]] = {}
    for route_id, office_id in office_map.items():
        result.setdefault(office_id, []).append(route_id)
    return result


def load_state(
    train_path: Path = DEFAULT_TRAIN_PATH,
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> AppState:
    """Load ML models and NCS scores. DB data is read per-request via sessions."""
    global _state

    models = load_models(models_dir)
    office_map = load_route_office_map(train_path)
    office_routes_map = _build_office_routes_map(office_map)

    ncs_scores, ncs_normalized = load_ncs(NCS_DEFAULT_PATH)
    ncs_allsteps, _ = load_ncs_allsteps()

    # Read initial settings from DB
    db = SessionLocal()
    try:
        from db.queries import get_all_settings
        settings = get_all_settings(db)
    finally:
        db.close()

    _state = AppState(
        models=models,
        train_path=train_path,
        office_map=office_map,
        office_routes_map=office_routes_map,
        confidence_level=float(settings.get("confidence_level", DEFAULT_CONFIDENCE_LEVEL)),
        ncs_scores=ncs_scores,
        ncs_normalized=ncs_normalized,
        ncs_allsteps=ncs_allsteps,
        granularity=float(settings.get("granularity", DEFAULT_GRANULARITY_HOURS)),
    )
    return _state
