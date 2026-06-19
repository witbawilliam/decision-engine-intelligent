from typing import Any, Dict, Optional

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
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
    model_name: str = Field(
        ...,
        example="auto_forecasting_model",
    )

    features: Dict[str, Any] = Field(
        ...,
        example={
            "feature_1": 12.5,
            "feature_2": 0.4,
        },
    )

    request_id: Optional[str] = None
    trace_id: Optional[str] = None

    include_explanations: bool = False

    # Counterfactual controls
    lever_col: Optional[str] = None
    target_goal: Optional[float] = None
    lever_min: Optional[float] = None
    lever_max: Optional[float] = None


# ==================================================
# SERVICE DEPENDENCY
# ==================================================
_prediction_service_instance: Optional[PredictionService] = None


def set_prediction_service(service: PredictionService) -> None:
    """
    Called during application startup.
    """
    global _prediction_service_instance
    _prediction_service_instance = service


def get_prediction_service() -> PredictionService:
    if _prediction_service_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service not initialized.",
        )

    return _prediction_service_instance


# ==================================================
# INFERENCE ENDPOINT
# ==================================================
@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
)
async def execute_model_inference(
    payload: InferencePayloadSchema,
    service: PredictionService = Depends(get_prediction_service),
):
    """
    Executes prediction + optional sensitivity analysis
    + manifold guard + counterfactual generation.
    """

    try:

        request_obj = PredictionRequest(
            model_name           = payload.model_name,
            features             = payload.features,
            include_explanations = payload.include_explanations,
            # Use UUID defaults if not provided
            **( {"request_id": payload.request_id} if payload.request_id else {} ),
            **( {"trace_id":   payload.trace_id}   if payload.trace_id   else {} ),
            # Counterfactual lever params — forwarded as-is (None is valid)
            lever_col   = payload.lever_col,
            target_goal = payload.target_goal,
            lever_min   = payload.lever_min,
            lever_max   = payload.lever_max,
        )

        response = await service.predict(request_obj)

        return response

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )

    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    except TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Prediction timeout.",
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(exc)}",
        )


# ==================================================
# HEALTH CHECK
# ==================================================
@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
)
async def health(
    service: PredictionService = Depends(get_prediction_service),
):
    return {
        "status": "healthy",
        "service": "prediction",
        "cache_models": len(service._model_cache),
    }