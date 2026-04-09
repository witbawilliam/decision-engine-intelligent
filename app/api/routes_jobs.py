
import logging
from fastapi import APIRouter, HTTPException, status
from datetime import datetime, timezone
from fastapi import Request

from app.schemas.job_schema import V1
from workers.celery_app import celery_app


router = APIRouter(prefix="/v1/train", tags=["Model Training"])
logger = logging.getLogger(__name__)



def _validate_job_create(request: V1.JobCreate) -> None:
    """
    Run the V1 field validators that are plain methods (not Pydantic @validators).
    Raises HTTPException(422) on any violation.
    """
    try:
        V1.validate_filename(request.filename)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    try:
        V1.validate_column_name(request.target_column)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))


def _dispatch(task_type: str, request: V1.JobCreate) -> V1.JobStatusResponse:
    """
    Shared dispatch helper.  Returns a JobStatusResponse on success.
    job_id is set to the request's idempotency_key so callers can poll by the
    same key they submitted (the Celery task UUID is an internal detail).
    """
    try:
        celery_app.dispatch_automl_task(task_type, request.model_dump())
    except ValueError as exc:
        # Unknown task_type — should not happen in normal flow, but guard anyway
        logger.error("Unknown task type '%s': %s", task_type, exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.error("Failed to dispatch '%s' task: %s", task_type, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error dispatching job",
        )

    return V1.JobStatusResponse(
        job_id=request.idempotency_key,   # stable, client-visible correlation ID
        status="queued",
        progress=0,
        updated_at=datetime.now(timezone.utc),
    )


@router.post(
    "/validate",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=V1.JobStatusResponse,
)
async def validate_dataset(request: V1.JobCreate):
    """
    Dispatches a dataset-validation job to workers.tasks_validation.task_validation.
    Maps to the 'validate' key in celery_app.dispatch_automl_task.
    """
    _validate_job_create(request)
    return _dispatch("validate", request)


@router.post(
    "/tabular",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=V1.JobStatusResponse,
)
async def train_tabular(request: V1.JobCreate):
    """
    Dispatches a tabular training job to workers.tasks_training.task_training.
    Rejects requests where problem_type is explicitly set to 'forecasting'
    (those belong on /forecast).
    """
    if request.problem_type == "forecasting":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Forecasting jobs must be submitted to /v1/train/forecast",
        )
    _validate_job_create(request)
    return _dispatch("train", request)


@router.post(
    "/forecast",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=V1.JobStatusResponse,
)
async def train_forecast(request: V1.JobCreate):
    """
    Dispatches a forecasting training job to workers.tasks_forecasting.task_forecasting.
    Requires problem_type == 'forecasting'.
    """
    if request.problem_type != "forecasting":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="problem_type must be 'forecasting' for this endpoint",
        )
    _validate_job_create(request)
    return _dispatch("forecast", request)