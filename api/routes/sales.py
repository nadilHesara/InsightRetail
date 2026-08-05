from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.schemas import DailySaleItem
from api.services import DatabaseService, get_database_url

router = APIRouter()


def get_service() -> DatabaseService:
    try:
        database_url = get_database_url()
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return DatabaseService(database_url)


@router.get("/sales/daily", response_model=list[DailySaleItem], tags=["Sales"])
def get_daily_sales(
    start_date: Optional[date] = Query(None, description="Earliest sale date to include"),
    end_date: Optional[date] = Query(None, description="Latest sale date to include"),
    limit: int = Query(30, ge=1, le=365, description="Maximum number of daily rows to return"),
    service: DatabaseService = Depends(get_service),
) -> list[DailySaleItem]:
    try:
        return service.fetch_daily_sales(start_date=start_date, end_date=end_date, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
