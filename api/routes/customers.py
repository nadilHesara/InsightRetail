from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.schemas import CustomerTopItem
from api.services import DatabaseService, get_database_url

router = APIRouter()


def get_service() -> DatabaseService:
    try:
        database_url = get_database_url()
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return DatabaseService(database_url)


@router.get("/customers/top", response_model=list[CustomerTopItem], tags=["Customers"])
def get_top_customers(
    start_date: Optional[date] = Query(None, description="Earliest order invoice date to include"),
    end_date: Optional[date] = Query(None, description="Latest order invoice date to include"),
    limit: int = Query(10, ge=1, le=100, description="Number of top customers to return"),
    service: DatabaseService = Depends(get_service),
) -> list[CustomerTopItem]:
    try:
        return service.fetch_top_customers(start_date=start_date, end_date=end_date, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
