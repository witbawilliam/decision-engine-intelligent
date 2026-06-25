from typing import Any, Dict, Optional

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    status,
)
from pydantic import BaseModel, Field

from service.prediction_service import (
    PredictionService,
    PredictionRequest,
    PredictionResponse,
)

router = APIRouter(
    prefix="/v1/inference",
    tags=["Inference Engine"],
)


# ==================================================
# REQUEST SCHEMA
# ==================================================
class InferencePayloadSchema(BaseModel):
    model_name: str = Field(..., example="auto_forecasting_model")
    features:   Dict[str, Any] = Field(..., example={"feature_1": 12.5})
    request_id: Optional[str]   = None
    trace_id:   Optional[str]   = None
    include_explanations: bool  = False
    lever_col:   Optional[str]   = None
    target_goal: Optional[float] = None
    lever_min:   Optional[float] = None
    lever_max:   Optional[float] = None


# ==================================================
# SERVICE DEPENDENCY — reads from app.state
# ==================================================
def get_prediction_service(request: Request) -> PredictionService:
    service = getattr(request.app.state, "prediction_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service not initialized.",
        )
    return service


# ==================================================
# INFERENCE ENDPOINT
# ==================================================
@router.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def execute_model_inference(
    payload: InferencePayloadSchema,
    request: Request,
    service: PredictionService = Depends(get_prediction_service),
):
    try:
        request_obj = PredictionRequest(
            model_name           = payload.model_name,
            features             = payload.features,
            include_explanations = payload.include_explanations,
            **( {"request_id": payload.request_id} if payload.request_id else {} ),
            **( {"trace_id":   payload.trace_id}   if payload.trace_id   else {} ),
            lever_col   = payload.lever_col,
            target_goal = payload.target_goal,
            lever_min   = payload.lever_min,
            lever_max   = payload.lever_max,
        )
        return await service.predict(request_obj)

    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except TimeoutError:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Prediction timeout.")
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference failed: {str(exc)}")


# ==================================================
# GET PREDICTIONS + ACTUALS (from model_evaluations)
# ==================================================
@router.get("/predictions/{model_name}", status_code=status.HTTP_200_OK)
async def get_model_predictions(
    model_name: str,
    request:    Request,
    version:    Optional[str] = Query(None),
    limit:      int           = Query(100),
    offset:     int           = Query(0),
    service:    PredictionService = Depends(get_prediction_service),
):
    try:
        return await service.get_model_predictions(
            model_name = model_name,
            version    = version,
            limit      = limit,
            offset     = offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch predictions: {str(exc)}")


# ==================================================
# GET AUDIT LOGS (from prediction_audit)
# ==================================================
@router.get("/audit/{model_name}", status_code=status.HTTP_200_OK)
async def get_prediction_audit(
    model_name: str,
    request:    Request,
    limit:      int = Query(100),
    offset:     int = Query(0),
    service:    PredictionService = Depends(get_prediction_service),
):
    try:
        return await service.get_prediction_audit(
            model_name = model_name,
            limit      = limit,
            offset     = offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch audit logs: {str(exc)}")


# ==================================================
# HEALTH CHECK
# ==================================================
@router.get("/health", status_code=status.HTTP_200_OK)
async def health(
    request: Request,
    service: PredictionService = Depends(get_prediction_service),
):
    return {
        "status":       "healthy",
        "service":      "prediction",
        "cache_models": len(service._model_cache),
    }