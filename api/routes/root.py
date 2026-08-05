from fastapi import APIRouter

router = APIRouter()


@router.get("/", tags=["Root"])
def root() -> dict[str, str]:
    return {
        "status": "ok",
        "message": "InsightRetail API is running. Visit /docs for API documentation.",
    }
