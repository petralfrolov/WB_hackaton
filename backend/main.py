"""WB Transport Optimizer — FastAPI entry point.

Run from the backend/ directory:
    uvicorn main:app --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from routers import (
    dispatch_router,
    health_router,
    optimize_router,
    settings_router,
    vehicles_router,
    warehouses_router,
)
from state import load_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models and config once at startup."""
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


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health_router)
app.include_router(warehouses_router)
app.include_router(dispatch_router)
app.include_router(optimize_router)
app.include_router(vehicles_router)
app.include_router(settings_router)
