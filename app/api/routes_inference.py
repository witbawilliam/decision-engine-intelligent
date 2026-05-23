import time
import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, HTTPException, status, Depends, Request

from service.prediction_service import PredictionService, PredictionRequest
from app.schemas.prediction_schema import PredictionRequest as APIRequest, PredictionResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/inference", tags=["Model Serving"])


def get_prediction_service(request: Request) -> PredictionService:
    """
    Production-safe dependency:
    Service is initialized once at app startup and reused.
    """
    return request.app.state.prediction_service


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Run ML inference with optional explanations"
)
async def predict(
    request: APIRequest,
    service: Annotated[PredictionService, Depends(get_prediction_service)],
):
    start_time = time.perf_counter()

    logger.info(
        "prediction_request_received",
        extra={
            "request_id": str(request.request_id),
            "model_name": request.model_name,
        },
    )

    try:
       
        service_request = PredictionRequest(
            features=request.features,
            model_name=request.model_name,
            request_id=str(request.request_id),
            trace_id=request.trace.trace_id,
            include_explanations=bool(request.include_explanations),
        )

        
        response = await service.predict(service_request)

        latency = (time.perf_counter() - start_time) * 1000

        logger.info(
            "prediction_success",
            extra={
                "request_id": str(request.request_id),
                "latency_ms": round(latency, 2),
                "cached": response.cached,
            },
        )

        return response

   
    except ValueError as e:
        logger.warning(
            "validation_error",
            extra={
                "request_id": str(request.request_id),
                "error": str(e),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    except asyncio.TimeoutError:
        logger.error(
            "inference_timeout",
            extra={"request_id": str(request.request_id)},
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Model inference timed out",
        )

    except Exception as e:
        logger.exception(
            "inference_internal_error",
            extra={
                "request_id": str(request.request_id),
                "error": str(e),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal inference error",
        )


@router.get("/health")
async def health(
    service: Annotated[PredictionService, Depends(get_prediction_service)],
):
    """
    Checks:
    - Redis
    - Postgres
    - Model Registry
    """
    status_map = service.health()

    if not all(status_map.values()):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=status_map,
        )

    return status_map