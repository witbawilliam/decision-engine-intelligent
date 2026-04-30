
import time
import logging
from fastapi import APIRouter, Depends, HTTPException, status
import asyncio
from typing import Annotated, Optional


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
    PredictionRequest as ServiceRequest,   
)
from core.models.model_registry import ModelRegistry
from core.pipelines.tabular_pipeline import TabularPipeline
import os
from storage.s3_client import S3Client

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/inference",
    tags=["Model Serving"]
)



_service_instance: Optional[PredictionService] = None


def get_prediction_service() -> PredictionService:
    global _service_instance
    if _service_instance is None:
        s3 = S3Client(
            bucket_name  = os.getenv("S3_DATASETS_BUCKET", "ml-datasets"),
            access_key   = os.getenv("AWS_ACCESS_KEY_ID"),
            secret_key   = os.getenv("AWS_SECRET_ACCESS_KEY"),
            region       = os.getenv("AWS_REGION", "us-east-1"),
            endpoint_url = os.getenv("S3__ENDPOINT_URL") or None,
        )
        registry = ModelRegistry(base_path="ml_registry", s3_client=s3)
        tabular_pipeline = registry.load("XGBoostModel")
        _service_instance = PredictionService(tabular_pipeline=tabular_pipeline)
    return _service_instance  



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
    start_time = time.perf_counter()
 
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
        trace_id      = request.trace.trace_id,
        pipeline_type = request.pipeline_type.value,
    )
 
    try:
        svc_response = await service.predict(service_request)
 
       
        sensitivity = None
        try:
            sensitivity = await service.analyze_sensitivity(
                service_request,
                pipeline_model=service._tabular,   
            )
        except Exception as exc:
            logger.warning(
                "sensitivity_analysis_skipped",
                extra={
                    "request_id": str(request.request_id),
                    "error":      str(exc),
                    "trace_id":   request.trace.trace_id,
                },
            )
 
        counterfactual = None
        try:
            counterfactual = await service.explain_counterfactual(
                service_request,
                pipeline_model=service._tabular,   
                training_data=None,                
                lever_col="Amount",               
                target_goal=0,                     
                bounds=(0, 50000),
            )
        except Exception as exc:
            logger.warning(
                "counterfactual_skipped",
                extra={
                    "request_id": str(request.request_id),
                    "error":      str(exc),
                    "trace_id":   request.trace.trace_id,
                },
            )
 
        risk = service.get_risk_score(service_request)
 
        latency_ms = (time.perf_counter() - start_time) * 1000
 
        return PredictionResponse(
            request_id    = request.request_id,
            trace         = request.trace,
            model_name    = svc_response.model_name,
            model_version = svc_response.model_version,
            status        = PredictionStatus.CACHED if svc_response.cached
                            else PredictionStatus.SUCCESS,
            prediction    = svc_response.prediction,
            cached        = svc_response.cached,
            latency_ms    = round(latency_ms, 2),
            explanations={
                "sensitivity":    sensitivity,
                "counterfactual": counterfactual,
                "manifold_risk":  risk,
            }
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
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,  
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