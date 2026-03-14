import polars as pl
from fastapi import APIRouter, HTTPException, Depends, status
from typing import Dict, Any, Annotated
from functools import lru_cache

from core.models.model_registry import ModelRegistry
from app.schemas.job_schema import V1
from core.contracts.problem_type import ProblemType

router = APIRouter(prefix="/v1/inference", tags=["ML Inference"])


@lru_cache(maxsize=32)
def get_cached_registry() -> ModelRegistry:
    return ModelRegistry()

@router.post(
    "/predict/{model_name}", 
    response_model=V1.JobResultResponse,
    status_code=status.HTTP_200_OK
)
async def predict(
    model_name: str,
    request: V1.JobCreate, 
    version: str = "latest",
    registry: ModelRegistry = Depends(get_cached_registry)
):
    """
    High-performance inference endpoint using Staff-level validation.
    """
    
    try:
        #  Optimized Model Loading
        model = registry.load(model_name, version=version if version != "latest" else None)
        
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "model_not_found", "message": f"Model {model_name} not available."}
            )

        
        input_dict = request.model_dump(exclude={"user_id", "idempotency_key", "problem_type"})
        input_data = pl.DataFrame([input_dict])

        prediction_result = model.predict(input_data)

        return V1.JobResultResponse(
            job_id=request.idempotency_key, 
            status="completed",
            model_version=getattr(model, "version", "1.0.0"),
            performance=getattr(model, "metrics_summary", {
                "primary_metric_name": "accuracy", 
                "primary_metric_value": 0.0
            }),
            insight_summary=f"Prediction generated successfully for user {request.user_id}"
        )

    except pl.exceptions.ComputeError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "data_error", "message": "Input data format is incompatible with model schema."}
        )
    except Exception as e:

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "internal_error", "message": "Critical failure in inference engine."}
        )