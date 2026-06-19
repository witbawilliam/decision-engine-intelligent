from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from feedback.feedback_service import FeedbackService
from feedback.error_logger import ErrorLogger

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/feedback",
    tags=["Feedback Engine"],
)


def normalize_model_identifier(model_name: str) -> str:
    """
    Ensures input strings from tracking paths are transformed into the clean
    database-compatible identifier before writing metrics.
    """
    normalized = model_name
    if "/" in model_name or ".parquet" in model_name:
        normalized = model_name.split("/")[-1]
        normalized = normalized.replace(".parquet_", "_")
        normalized = normalized.replace(".parquet", "")
    return normalized




class FeedbackRequest(BaseModel):
    """
    Body structure for incoming ground-truth feedback payloads.
    """
    model_name: str                      = Field(..., min_length=1, max_length=128, example="auto_regression_model")
    prediction: float                    = Field(..., description="Value the model predicted.")
    actual:     float                    = Field(..., description="Real observed value.")
    metadata:   Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional context — job_id, user_id, input snapshot, etc.",
    )


class FeedbackResponse(BaseModel):
    """Validated API response containing calculated performance statistics."""
    feedback_id:    str
    model_name:     str
    prediction:     float
    actual:         float
    absolute_error: float
    squared_error:  float
    relative_error: Optional[float]  = None
    recorded_at:    datetime
    metadata:       Dict[str, Any]   = Field(default_factory=dict)



_feedback_service_instance: Optional[FeedbackService] = None

def get_feedback_service() -> FeedbackService:
    global _feedback_service_instance
    if _feedback_service_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feedback collection service is uninitialized."
        )
    return _feedback_service_instance


# --- UTILITY HELPERS ---

def _compute_metrics(prediction: float, actual: float) -> dict[str, float | None]:
    """Computes regression error metrics cleanly for downstream dashboards."""
    abs_error = abs(prediction - actual)
    sq_error  = (prediction - actual) ** 2
    rel_error = abs_error / abs(actual) if actual != 0 else None
    return {
        "absolute_error": abs_error,
        "squared_error":  sq_error,
        "relative_error": rel_error,
    }


def _build_response(
    feedback_id: str,
    payload:     FeedbackRequest,
    metrics:     dict[str, Any],
    recorded_at: datetime,
    clean_name:  str,
) -> FeedbackResponse:
    """Constructs the FeedbackResponse object output."""
    return FeedbackResponse(
        feedback_id    = feedback_id,
        model_name     = clean_name,
        prediction     = payload.prediction,
        actual         = payload.actual,
        absolute_error = metrics["absolute_error"],
        squared_error  = metrics["squared_error"],
        relative_error = metrics["relative_error"],
        recorded_at    = recorded_at,
        metadata       = payload.metadata or {},
    )




@router.post(
    "/",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit model-prediction feedback",
    description="Records ground-truth actual values against previous predictions for drift monitoring.",
)
async def submit_feedback(
    request: Request,
    payload: FeedbackRequest,
    service: FeedbackService = Depends(get_feedback_service)
) -> FeedbackResponse:
    
    feedback_id = str(uuid.uuid4())
    recorded_at = datetime.now(tz=timezone.utc)
    clean_model_name = normalize_model_identifier(payload.model_name)

    try:
        metrics = _compute_metrics(payload.prediction, payload.actual)

        
        service.store_feedback(
            model_id     = clean_model_name,
            input_data   = payload.metadata or {},
            prediction   = payload.prediction,
            actual_value = payload.actual,
        )

        logger.info(
            "feedback_recorded",
            extra={
                "feedback_id":    feedback_id,
                "model_name":     clean_model_name,
                "absolute_error": round(metrics["absolute_error"], 6),
                "client":         request.client.host if request.client else "unknown",
            },
        )

        return _build_response(feedback_id, payload, metrics, recorded_at, clean_model_name)

    except Exception as exc:
        ErrorLogger.log_error(
            component = "routes_feedback/submit",
            error     = exc,
            context   = {
                "feedback_id": feedback_id,
                "model_name":  clean_model_name,
                "prediction":  payload.prediction,
                "actual":      payload.actual,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record feedback metrics.",
        ) from exc


@router.get(
    "/{model_name}",
    response_model=List[Dict[str, Any]],
    status_code=status.HTTP_200_OK,
    summary="Retrieve feedback for a model",
    description="Returns all logged historical performance metrics for a specific production model architecture.",
)
async def get_feedback(
    model_name: str,
    service: FeedbackService = Depends(get_feedback_service)
) -> List[Dict[str, Any]]:
    
    clean_model_name = normalize_model_identifier(model_name)
    try:
        records = service.get_feedback(model_id=clean_model_name)

        logger.info(
            "feedback_retrieved",
            extra={
                "model_name":   clean_model_name,
                "record_count": len(records),
            },
        )

        return records

    except Exception as exc:
        ErrorLogger.log_error(
            component = "routes_feedback/get_feedback",
            error     = exc,
            context   = {"model_name": clean_model_name},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve feedback for model '{clean_model_name}'.",
        ) from exc