from fastapi import APIRouter, Depends, HTTPException

from api.schemas import PredictRequest, PredictResponse
from api.services import build_prediction, get_database_url, load_model

router = APIRouter()

MODEL = load_model()


def get_service() -> str:
    return get_database_url()


@router.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(
    request_data: PredictRequest,
    _: str = Depends(get_service),
) -> PredictResponse:
    try:
        prediction = build_prediction(request_data.dict(), MODEL)
        return PredictResponse(**prediction)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
