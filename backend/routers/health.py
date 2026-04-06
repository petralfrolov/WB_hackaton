from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    """Return a simple liveness probe response."""
    return {"status": "ok"}
