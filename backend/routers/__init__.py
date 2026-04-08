from .health import router as health_router
from .vehicles import router as vehicles_router
from .settings import router as settings_router
from .warehouses import router as warehouses_router
from .optimize import router as optimize_router
from .dispatch import router as dispatch_router
from .call import router as call_router
from .metrics import router as metrics_router

__all__ = [
    "health_router",
    "vehicles_router",
    "settings_router",
    "warehouses_router",
    "optimize_router",
    "dispatch_router",
    "call_router",
    "metrics_router",
]
