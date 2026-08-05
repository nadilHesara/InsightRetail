from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.schemas import ProductTopItem
from api.services import DatabaseService, get_database_url

router = APIRouter()


def get_service() -> DatabaseService:
    try:
        database_url = get_database_url()
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return DatabaseService(database_url)


@router.get("/products/top", response_model=list[ProductTopItem], tags=["Products"])
def get_top_products(
    start_date: Optional[date] = Query(None, description="Earliest order invoice date to include"),
    end_date: Optional[date] = Query(None, description="Latest order invoice date to include"),
    limit: int = Query(10, ge=1, le=100, description="Number of top products to return"),
    service: DatabaseService = Depends(get_service),
) -> list[ProductTopItem]:
    try:
        return service.fetch_top_products(start_date=start_date, end_date=end_date, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
