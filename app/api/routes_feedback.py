from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from feedback.feedback_service import FeedbackService


from feedback.error_logger import ErrorLogger

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/feedback",
    tags=["Feedback"],
)




class FeedbackRequest(BaseModel):
    """
    Body for POST /feedback/.

    Fields map to FeedbackService.store_feedback() parameters:
        model_name  → model_id
        metadata    → input_data
        actual      → actual_value
    """
    model_name: str                      = Field(..., min_length=1, max_length=128)
    prediction: float                    = Field(..., description="Value the model predicted.")
    actual:     float                    = Field(..., description="Real observed value.")
    metadata:   Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional context — job_id, user_id, input snapshot, etc.",
    )


class FeedbackResponse(BaseModel):
    """Response returned after POST /feedback/."""
    feedback_id:    str
    model_name:     str
    prediction:     float
    actual:         float
    absolute_error: float
    squared_error:  float
    relative_error: Optional[float]  = None
    recorded_at:    datetime
    metadata:       Dict[str, Any]   = Field(default_factory=dict)




def _compute_metrics(prediction: float, actual: float) -> dict[str, float | None]:
    """
    Computes regression error metrics.
    Called before FeedbackService.store_feedback() so metrics are available
    in both the response body and any downstream monitoring.
    """
    abs_error = abs(prediction - actual)
    sq_error  = (prediction - actual) ** 2
    rel_error: float | None = abs_error / abs(actual) if actual != 0 else None
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
) -> FeedbackResponse:
    """Constructs the FeedbackResponse from the stored payload and computed metrics."""
    return FeedbackResponse(
        feedback_id    = feedback_id,
        model_name     = payload.model_name,
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
    description=(
        "Records ground-truth actuals against model predictions. "
        "Stored in postgres model_feedback table for drift monitoring and retraining."
    ),
)
async def submit_feedback(
    request: Request,
    payload: FeedbackRequest,
) -> FeedbackResponse:
    """
    Connection map
    
    FeedbackRequest  (inline Pydantic — no job_schema)
        validates model_name, prediction, actual, metadata

    _compute_metrics(prediction, actual)
        absolute_error, squared_error, relative_error

    FeedbackService.store_feedback()          ← feedback_service.py
         model_id     = payload.model_name
         input_data   = payload.metadata or {}
         prediction   = payload.prediction
         actual_value = payload.actual
         PostgresClient.insert("model_feedback", record)

    ErrorLogger.log_error()                   ← error_logger.py
         called on any exception
         component = "routes_feedback/submit"
         context includes feedback_id, model_name, prediction, actual
    """
    feedback_id = str(uuid.uuid4())
    recorded_at = datetime.now(tz=timezone.utc)

    try:
        
        metrics = _compute_metrics(payload.prediction, payload.actual)

        
        FeedbackService.store_feedback(
            model_id     = payload.model_name,
            input_data   = payload.metadata or {},
            prediction   = payload.prediction,
            actual_value = payload.actual,
        )

        logger.info(
            "feedback_recorded",
            extra={
                "feedback_id":    feedback_id,
                "model_name":     payload.model_name,
                "absolute_error": round(metrics["absolute_error"], 6),
                "client":         request.client.host if request.client else "unknown",
            },
        )

        
        return _build_response(feedback_id, payload, metrics, recorded_at)

    except Exception as exc:
        
        ErrorLogger.log_error(
            component = "routes_feedback/submit",
            error     = exc,
            context   = {
                "feedback_id": feedback_id,
                "model_name":  payload.model_name,
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
    description=(
        "Returns all feedback records for the given model from postgres, "
        "ordered by timestamp descending. Returns empty list if none exist."
    ),
)
async def get_feedback(model_name: str) -> List[Dict[str, Any]]:
    """
    Connection map

    model_name path param
         FeedbackService.get_feedback(model_id=model_name)    feedback_service.py
             SELECT * FROM model_feedback WHERE model_id = %s
             returns List[Dict] — empty list is valid, not a 404

    ErrorLogger.log_error()                                       error_logger.py
         called on any exception
         component = "routes_feedback/get_feedback"
    """
    try:
        records = FeedbackService.get_feedback(model_id=model_name)

        logger.info(
            "feedback_retrieved",
            extra={
                "model_name":   model_name,
                "record_count": len(records),
            },
        )

        return records

    except Exception as exc:
        ErrorLogger.log_error(
            component = "routes_feedback/get_feedback",
            error     = exc,
            context   = {"model_name": model_name},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve feedback for model '{model_name}'.",
        ) from exc