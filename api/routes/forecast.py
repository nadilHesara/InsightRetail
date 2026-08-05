from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.schemas import ForecastItem
from api.services import DatabaseService, get_database_url

router = APIRouter()


def get_service() -> DatabaseService:
    try:
        database_url = get_database_url()
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return DatabaseService(database_url)


@router.get("/forecast", response_model=list[ForecastItem], tags=["Forecast"])
def get_forecast(
    start_date: Optional[date] = Query(None, description="Earliest forecast date to include"),
    end_date: Optional[date] = Query(None, description="Latest forecast date to include"),
    limit: int = Query(30, ge=1, le=365, description="Number of forecast rows to return"),
    service: DatabaseService = Depends(get_service),
) -> list[ForecastItem]:
    try:
        return service.fetch_forecast(start_date=start_date, end_date=end_date, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
