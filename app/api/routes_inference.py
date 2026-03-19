import time
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
import asyncio


from app.schemas.prediction_schema import (
    PredictionRequest,
    PredictionResponse,
    PipelineType,
    PredictionStatus,
    PredictionError,
    ErrorCode,
    TraceContext,
)


from service.prediction_service import (
    PredictionService,
    PredictionRequest as ServiceRequest,   # internal dataclass ≠ schema Pydantic model
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/inference",
    tags=["Model Serving"]
)



def get_prediction_service() -> PredictionService:
    """
    Provides a PredictionService instance.

    Wire real pipeline / storage dependencies here.
    Replace with mocks in tests by overriding this dependency.
    """
    return PredictionService()          # pipelines + storage injected in __init__



@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute Model Inference",
    description=(
        "Validates input via PredictionRequest schema, "
        "checks Redis cache, dispatches to the correct pipeline, "
        "enriches via decision_engine, and returns a PredictionResponse."
    ),
)
async def predict(
   
    request: PredictionRequest,
    service: Annotated[PredictionService, Depends(get_prediction_service)],
):
    """
    Single-item inference endpoint.

    Connection map
    
    prediction_schema.PredictionRequest
         validated by FastAPI before this function body runs
         .cache_key    - pre-computed SHA-256 (used by service._cache_get)
         .trace          -TraceContext forwarded into service for structured logs
         .pipeline_type - routed to tabular_pipeline or temporal_pipeline

    prediction_service.PredictionService.predict()
        checks RedisClient cache (cache hit → skips pipeline entirely)
        dispatches to TemporalPipeline or TabularPipeline
        runs decision_engine enrichment (counterfactuals, sensitivity)
        writes audit record to PostgresClient
        records latency via MetricsRegistry
        returns internal PredictionResponse dataclass

    This route then maps the service dataclass → schema PredictionResponse
    so the response matches the declared response_model exactly.
    """
    start_time = time.perf_counter()

    # Log incoming metadata — never log raw feature values (PII risk)
    logger.info(
        "prediction_initiated",
        extra={
            "model_name":    request.model_name,
            "request_id":    str(request.request_id),
            "pipeline_type": request.pipeline_type,
            "trace_id":      request.trace.trace_id,
        },
    )

    
    service_request = ServiceRequest(
        features      = request.features,
        model_name    = request.model_name,
        request_id    = str(request.request_id),
        trace_id      = request.trace.trace_id,   # propagate trace across service boundary
        pipeline_type = request.pipeline_type.value,  # enum → "tabular" / "temporal"
    )

    try:
        
        svc_response = await service.predict(service_request)

        latency_ms = (time.perf_counter() - start_time) * 1000

        
        return PredictionResponse(
            request_id    = request.request_id,
            trace         = request.trace,                  # full TraceContext forwarded
            model_name    = svc_response.model_name,
            model_version = svc_response.model_version,
            status        = PredictionStatus.CACHED if svc_response.cached
                            else PredictionStatus.SUCCESS,
            prediction    = svc_response.prediction,
            cached        = svc_response.cached,
            latency_ms    = round(latency_ms, 2),
        )

    
    except ValueError as ve:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.warning(
            "inference_validation_failed",
            extra={
                "request_id": str(request.request_id),
                "error":      str(ve),
                "trace_id":   request.trace.trace_id,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(ve),
        )

   
    
    except asyncio.TimeoutError as te:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "inference_timeout",
            extra={
                "request_id": str(request.request_id),
                "error":      str(te),
                "latency_ms": round(latency_ms, 2),
                "trace_id":   request.trace.trace_id,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Inference timed out. Try again or reduce feature count.",
        )

    
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "inference_critical_error",
            extra={
                "request_id": str(request.request_id),
                "error_type": type(e).__name__,
                "error":      str(e),
                "latency_ms": round(latency_ms, 2),
                "trace_id":   request.trace.trace_id,
            },
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing the prediction.",
        )


# GET /inference/health 
@router.get(
    "/health",
    summary="Service Health Check",
    description="Pings Redis and PostgreSQL via PredictionService.health(). Used by load balancers.",
)
def health(
    service: Annotated[PredictionService, Depends(get_prediction_service)],
):
    """
    Calls PredictionService.health() which pings:
      - redis_client.ping()    → checks feature cache + Celery broker
      - postgres_client.ping() → checks audit log DB

    Returns {"redis": True/False, "postgres": True/False}
    """
    return service.health()