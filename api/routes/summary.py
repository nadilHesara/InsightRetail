from fastapi import APIRouter, Depends, HTTPException

from api.schemas import SummaryResponse
from api.services import DatabaseService, get_database_url

router = APIRouter()


def get_service() -> DatabaseService:
    try:
        database_url = get_database_url()
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return DatabaseService(database_url)


@router.get("/summary", response_model=SummaryResponse, tags=["Summary"])
def get_summary(service: DatabaseService = Depends(get_service)) -> SummaryResponse:
    try:
        return service.fetch_summary()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
