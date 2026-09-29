import logging
from fastapi import APIRouter, HTTPException, status
from datetime import datetime, timezone
from celery.result import AsyncResult

from app.schemas.job_schema import V1
from workers.celery_app import celery_app
from storage.postgres_client import PostgresClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/train", tags=["Model Training"])

# Generic job-status polling, shared by training AND forecast batch jobs --
# both write to the same `jobs` table via PostgresClient.upsert, so this
# endpoint doesn't need to know or care which task type created the job.
jobs_router = APIRouter(prefix="/v1/jobs", tags=["Jobs"])

_CELERY_STATE_MAP = {
    "PENDING":  "queued",
    "STARTED":  "running",
    "SUCCESS":  "completed",
    "FAILURE":  "failed",
    "REVOKED":  "cancelled",
    "RETRY":    "running",
}


def _validate_job_create(request: V1.JobCreate) -> None:
    try:
        V1.validate_filename(request.filename)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    try:
        V1.validate_column_name(request.target_column)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))


def _dispatch(task_type: str, request: V1.JobCreate) -> V1.JobStatusResponse:
    try:
        PostgresClient.upsert(
            table="jobs",
            data={
                "id": request.idempotency_key,
                "status": "queued",
                "progress": 0,
                "updated_at": datetime.now(timezone.utc),
            },
            conflict_columns=["id"],
        )
    except Exception as exc:
        logger.error("Failed to create job record: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create job record")

    try:
        celery_app.dispatch_automl_task(task_type, request.model_dump())
    except ValueError as exc:
        PostgresClient.execute("DELETE FROM jobs WHERE id = %s", (request.idempotency_key,))
        logger.error("Failed to dispatch task, cleaned up DB: %s", exc)
        raise HTTPException(status_code=500, detail="Job dispatch failed")

    return V1.JobStatusResponse(
        job_id=request.idempotency_key,
        status="queued",
        progress=0,
        updated_at=datetime.now(timezone.utc),
    )


@jobs_router.get(
    "/status/{job_id:path}",
    response_model=V1.JobStatusResponse,
)
async def get_job_status(job_id: str):
    """
    Poll the status of any dispatched job by its idempotency_key / job_id --
    training, forecast batch, or any future job type. Checks the DB first
    (most accurate, source of truth for progress %), falls back to Celery's
    result backend if the DB lookup fails or the row doesn't exist yet.
    """
    try:
        rows = PostgresClient.query(
            sql_str="SELECT * FROM jobs WHERE id = %s",
            params=(job_id,),
        )
        row = rows[0] if rows else None
    except Exception as exc:
        logger.warning("DB status lookup failed for %s: %s", job_id, exc)
        row = None

    if row:
        return V1.JobStatusResponse(
            job_id=job_id,
            status=row["status"],
            progress=row.get("progress", 0),
            updated_at=row.get("updated_at").isoformat()
                if hasattr(row.get("updated_at"), "isoformat")
                else datetime.now(timezone.utc),
        )

    try:
        celery_result = AsyncResult(job_id, app=celery_app)
        if celery_result.state == "PENDING" and not row:
            raise HTTPException(status_code=404, detail="Job ID not found in system")

        celery_status = _CELERY_STATE_MAP.get(celery_result.state, "queued")
        meta = celery_result.info or {}
        progress = meta.get("progress", 0) if isinstance(meta, dict) else 0

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Celery status lookup failed for %s: %s", job_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve job status",
        )

    return V1.JobStatusResponse(
        job_id=job_id,
        status=celery_status,
        progress=progress,
        updated_at=datetime.now(timezone.utc),
    )


@router.post("/validate", status_code=status.HTTP_202_ACCEPTED, response_model=V1.JobStatusResponse)
async def validate_dataset(request: V1.JobCreate):
    if not request.s3_key:
        raise HTTPException(status_code=422, detail="s3_key is required for validation")
    if not request.reference_s3_key:
        raise HTTPException(status_code=422, detail="reference_s3_key is required for validation")

    _validate_job_create(request)
    return _dispatch("validate", request)


@router.post("/tabular", status_code=status.HTTP_202_ACCEPTED, response_model=V1.JobStatusResponse)
async def train_tabular(request: V1.JobCreate):
    if request.problem_type == "forecasting":
        raise HTTPException(status_code=400, detail="Forecasting jobs must be submitted to /v1/train/forecast")
    _validate_job_create(request)
    return _dispatch("train", request)


@router.post("/forecast", status_code=status.HTTP_202_ACCEPTED, response_model=V1.JobStatusResponse)
async def train_forecast(request: V1.JobCreate):
    if request.problem_type != "forecasting":
        raise HTTPException(status_code=400, detail="problem_type must be 'forecasting' for this endpoint")
    _validate_job_create(request)
    return _dispatch("forecast", request)