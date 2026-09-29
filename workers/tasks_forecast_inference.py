from __future__ import annotations

import logging
from datetime import datetime, timezone
from celery import Task, shared_task

from storage.postgres_client import PostgresClient
from storage.s3_client import S3Client
from core.models.model_registry import ModelRegistry
from service.forecast_service import PredictionService

logger = logging.getLogger(__name__)


def _build_s3_client() -> S3Client:
    """Constructs S3Client from settings -- identical to how the training
    task builds it, so both use the same bucket/credentials/region."""
    from app.config import get_settings
    settings = get_settings()
    return S3Client(
        bucket_name=settings.s3.bucket_name,
        endpoint_url=settings.s3.endpoint_url,
        access_key=settings.s3.access_key,
        secret_key=settings.s3.secret_key,
        region=settings.s3.region,
    )


def _report_progress(task, job_id: str, stage: str, percent: int) -> None:
    """Updates Celery's own task state (visible via AsyncResult.info) and
    the jobs table's progress column. Two independent try/excepts so a
    Celery state-update failure never blocks the Postgres write, or vice
    versa -- progress reporting is best-effort and must never abort the
    actual forecasting work."""
    try:
        task.update_state(state="PROGRESS", meta={"stage": stage, "percent": percent})
    except Exception:
        logger.warning("Failed to update Celery task state for job_id=%s", job_id)

    try:
        PostgresClient.upsert(
            table="jobs",
            data={
                "id":         job_id,
                "status":     "running",
                "progress":   percent,
                "updated_at": datetime.now(timezone.utc),
            },
            conflict_columns=["id"],
        )
    except Exception:
        logger.warning("Failed to update jobs.progress for job_id=%s", job_id)


def _build_error_result(job_id: str, message: str) -> dict:
    """Standard error-shaped return value for a failed Celery task,
    matching the {'status': 'ERROR', ...} shape seen in existing task logs."""
    return {
        "status":     "ERROR",
        "job_id":     job_id,
        "message":    message,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    }


@shared_task(
    bind=True,
    name="workers.tasks_forecast_inference.run_batch_forecast_task",
    queue="inference",
    time_limit=900,
    soft_time_limit=840,)
def run_batch_forecast_task(self, job_payload: dict) -> dict:
   
    task_id = self.request.id
    job_id  = job_payload.get("idempotency_key") or task_id

    # Mark job running immediately -- visible before any per-product work starts.
    PostgresClient.upsert(
        table="jobs",
        data={
            "id":         job_id,
            "status":     "running",
            "progress":   0,
            "updated_at": datetime.now(timezone.utc),
        },
        conflict_columns=["id"],
    )

    try:
        s3 = _build_s3_client()
        registry = ModelRegistry(s3_client=s3)

        
        service = PredictionService(
            registry=registry,
            redis=None,
            pg=None,
            default_forecast_horizon=job_payload.get("forecast_horizon", 30),
        )

        _report_progress(self, job_id, "running_batch_forecast", 20)

        
        results = service.run_batch_forecast(
            periods=job_payload.get("forecast_horizon"),
        )

        succeeded = [pid for pid, r in results.items() if r["status"] == "success"]
        failed    = [pid for pid, r in results.items() if r["status"] == "failed"]

        _report_progress(self, job_id, "persisting_results", 70)

        rows_written = registry.save_forecast_results(results, job_id=job_id)

        # Build the full response BEFORE marking the job completed -- avoids
        # writing "completed" to the jobs table before we know the response
        # payload itself is valid (same ordering fix applied to training).
        response = {
            "status":            "SUCCESS",
            "task_id":           task_id,
            "job_id":            job_id,
            "products_forecast": succeeded,
            "products_failed":   failed,
            "total_products":    len(results),
            "rows_written":      rows_written,
            "timestamp":         datetime.now(timezone.utc).isoformat(),
        }

        PostgresClient.upsert(
            table="jobs",
            data={
                "id":         job_id,
                "status":     "completed",
                "progress":   100,
                "updated_at": datetime.now(timezone.utc),
            },
            conflict_columns=["id"],
        )

        logger.info(
            "Batch forecast job %s complete — %d/%d products succeeded, %d rows written.",
            job_id, len(succeeded), len(results), rows_written,
        )

        return response

    except Exception as e:
        PostgresClient.upsert(
            table="jobs",
            data={
                "id":             job_id,
                "status":         "failed",
                "progress":       0,
                "updated_at":     datetime.now(timezone.utc),
                "error_message":  str(e),
            },
            conflict_columns=["id"],
        )
        logger.exception("Batch forecast job failed", extra={"job_id": job_id})
        return _build_error_result(job_id, str(e))