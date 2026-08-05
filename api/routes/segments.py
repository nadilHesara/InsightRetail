from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.schemas import SegmentItem
from api.services import DatabaseService, get_database_url

router = APIRouter()


def get_service() -> DatabaseService:
    try:
        database_url = get_database_url()
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return DatabaseService(database_url)


@router.get("/segments", response_model=list[SegmentItem], tags=["Segments"])
def get_segments(
    limit: int = Query(50, ge=1, le=500, description="Maximum number of customer segments to return"),
    service: DatabaseService = Depends(get_service),
) -> list[SegmentItem]:
    try:
        return service.fetch_segments(limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
