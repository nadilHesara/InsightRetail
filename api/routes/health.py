from fastapi import APIRouter

from api.schemas import HealthCheckResponse

router = APIRouter()


@router.get("/health", response_model=HealthCheckResponse, tags=["Health"])
def health_check() -> HealthCheckResponse:
    return HealthCheckResponse()
