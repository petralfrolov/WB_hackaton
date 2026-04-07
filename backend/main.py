"""WB Transport Optimizer — FastAPI entry point.

Run from the backend/ directory:
    uvicorn main:app --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from routers import (
    dispatch_router,
    health_router,
    optimize_router,
    settings_router,
    vehicles_router,
    warehouses_router,
    call_router,
)
from core.state import load_state, get_state
from db.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise DB and load ML models."""
    init_db()
    load_state()
    yield


app = FastAPI(
    title="WB Transport Optimizer",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global error handler ─────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}: {exc}"},
    )


# ── Middleware: block mutating requests while dispatches are running ──────────
# Dispatch and read endpoints are always allowed.
_READ_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
_DISPATCH_PATHS = {"/dispatch", "/optimize", "/call"}


class BlockWritesDuringDispatchMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method not in _READ_SAFE_METHODS and request.url.path not in _DISPATCH_PATHS:
            try:
                state = get_state()
                if state.dispatching:
                    return JSONResponse(
                        status_code=423,
                        content={"detail": "Пожалуйста, дождитесь окончания расчётов по маршрутам."},
                    )
            except RuntimeError:
                pass  # state not loaded yet
        return await call_next(request)


app.add_middleware(BlockWritesDuringDispatchMiddleware)


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health_router)
app.include_router(warehouses_router)
app.include_router(dispatch_router)
app.include_router(optimize_router)
app.include_router(vehicles_router)
app.include_router(settings_router)
app.include_router(call_router)
