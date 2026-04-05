from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    """Return a simple liveness probe response."""
    return {"status": "ok"}
