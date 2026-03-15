from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

# Import the V1 namespace container
from app.schemas.job_schema import V1

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/feedback",
    tags=["Feedback"],
)

def _compute_metrics(prediction: float, actual: float) -> dict[str, float | None]:
    """
    Computes regression error metrics. 
    In an enterprise system, this logic might eventually move to a core 'metrics' utility.
    """
    abs_error = abs(prediction - actual)
    sq_error = (prediction - actual) ** 2
    rel_error: float | None = (
        abs_error / abs(actual) if actual != 0 else None
    )
    return {
        "absolute_error": abs_error,
        "squared_error": sq_error,
        "relative_error": rel_error,
    }

def _build_response(
    feedback_id: str,
    payload: V1.FeedbackRequest,
    metrics: dict[str, Any],
    recorded_at: datetime,
) -> V1.FeedbackResponse:
    """Constructs the versioned response object."""
    return V1.FeedbackResponse(
        feedback_id=feedback_id,
        model_name=payload.model_name,
        prediction=payload.prediction,
        actual=payload.actual,
        absolute_error=metrics["absolute_error"],
        squared_error=metrics["squared_error"],
        relative_error=metrics["relative_error"],
        recorded_at=recorded_at,
        metadata=payload.metadata or {},
    )

@router.post(
    "/",
    response_model=V1.FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit model-prediction feedback",
)
async def submit_feedback(
    request: Request,
    payload: V1.FeedbackRequest, # Correctly referencing the nested class
) -> V1.FeedbackResponse:
    """
    Record ground-truth (actuals) against model predictions to monitor performance drift.
    """
    feedback_id = str(uuid.uuid4())
    recorded_at = datetime.now(tz=timezone.utc)

    try:
        # Compute Metrics
        metrics = _compute_metrics(payload.prediction, payload.actual)
        
        #  Map to V1 Response Schema
        response = _build_response(feedback_id, payload, metrics, recorded_at)
        
        #  Structured Logging
        logger.info(
            "Feedback recorded",
            extra={
                "feedback_id": feedback_id,
                "model": payload.model_name,
                "abs_err": metrics["absolute_error"],
                "client": request.client.host if request.client else "unknown",
            },
        )
        return response

    except Exception as exc:
        logger.exception(f"Feedback processing failed: {feedback_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record feedback metrics."
        ) from exc