from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
import os

import polars as pl
from celery import Task, shared_task
from celery.exceptions import MaxRetriesExceededError

from core.pipelines.temporal_pipeline import TemporalPipeline
from monitoring.metrics import track_training_latency
from datetime import datetime, timezone
from app.schemas.job_schema import V1

logger = logging.getLogger(__name__)



STATUS_SUCCESS = "SUCCESS"
STATUS_FAILED  = "FAILED"   
STATUS_ERROR   = "ERROR"    

TaskResult = dict[str, Any]




@dataclass
class _ProgressReporter:
    """
    Wraps ``self.update_state`` to standardise the progress payload shape
    and emit a structured debug log on every step change.
    """

    task: Task
    task_id: str

    def report(self, step: str, percent: int) -> None:
        self.task.update_state(
            state="PROGRESS",
            meta={"step": step, "percent": percent, "task_id": self.task_id},
        )
        logger.debug(
            "Forecasting task progress",
            extra={"task_id": self.task_id, "step": step, "percent": percent},
        )


def submit_job(payload: dict) -> str:
    job = V1.JobCreate(**payload)
    
    # Resolve storage path via env var (Defaulting to /data/ for local dev)

    base_storage = os.getenv("STORAGE_ROOT", "/data")
    full_path = os.path.join(base_storage, job.filename)
    
    task = train_forecasting_task.apply_async(
        kwargs={
            "dataset_path": full_path,
            "time_column": "timestamp", 
            "target_column": job.target_column,
        },
        task_id=job.idempotency_key,            
    )
    return task.id




def _success(task_id: str, result: Any, freq: str) -> TaskResult:
    response = V1.JobResultResponse(
        job_id=task_id,
        status="completed",
        model_version=result.model_version,
        performance={
            "mae": result.metrics.get("mae"),
            "rmse": result.metrics.get("rmse"),
            "horizon": result.horizon,
            "freq": freq,
        },
        insight_summary=f"Forecasting complete for {freq} frequency.",
    )
    payload = response.model_dump()
    payload["status"] = STATUS_SUCCESS  # Explicitly add the internal status flag
    return payload

def _failure(task_id: str, reason: str) -> TaskResult:
    return {"status": STATUS_FAILED, "task_id": task_id, "reason": reason}


def _error(task_id: str, message: str) -> TaskResult:
    return {"status": STATUS_ERROR, "task_id": task_id, "message": message}




class ForecastingTask(Task):
    """
    Celery base for time-series forecasting jobs.

    Provides structured failure/retry logging and a telemetry hook via
    ``track_training_latency`` so dashboards stay accurate without log parsing.
    """

    abstract = True  # Not registered as a concrete task.

    def on_failure(
        self,
        exc: Exception,
        task_id: str,
        args: tuple,
        kwargs: dict,
        einfo: Any,
    ) -> None:
        logger.error(
            "Forecasting task permanently failed",
            extra={
                "task_id": task_id,
                "exc_type": type(exc).__name__,
                "exc_message": str(exc),
                # args[0] is dataset_path — safe metadata, no PII.
                "dataset_path": args[0] if args else kwargs.get("dataset_path"),
            },
            exc_info=einfo,
        )
        track_training_latency(task_id=task_id, status="failed", pipeline="forecasting")

    def on_retry(
        self,
        exc: Exception,
        task_id: str,
        args: tuple,
        kwargs: dict,
        einfo: Any,
    ) -> None:
        logger.warning(
            "Forecasting task retrying",
            extra={
                "task_id": task_id,
                "exc_type": type(exc).__name__,
                "attempt": self.request.retries + 1,
                "max_retries": self.max_retries,
            },
        )

@dataclass
class _ProgressReporter:
    task: Task
    task_id: str

    def report(self, step: str, percent: int) -> None:
        status_payload = V1.JobStatusResponse(
            job_id=self.task_id,
            status="running",
            progress=percent,
            updated_at=datetime.now(timezone.utc),
        )
        self.task.update_state(
            state="PROGRESS",
            meta=status_payload.model_dump(),   # validated dict
        )


@shared_task(
    bind=True,
    base=ForecastingTask,
    name="workers.tasks.train_forecasting_task",
    queue="training_queue",
    time_limit=7200,            
    soft_time_limit=7140,       
    autoretry_for=(OSError, IOError),
    dont_autoretry_for=(FileNotFoundError, MemoryError),
    max_retries=2,              
    retry_backoff=True,         
    retry_backoff_max=120,
    retry_jitter=True,          
)
def train_forecasting_task(
    self,
    dataset_path: str,
    time_column: str,
    target_column: str,
    freq: str = "H",
) -> TaskResult:
    """
    Train a time-series forecasting model on the dataset at ``dataset_path``.

    Parameters
    ----------
    dataset_path:
        Absolute path to the validated CSV file on shared storage.
    time_column:
        Name of the datetime column used as the time index.
    target_column:
        Name of the column the model should forecast.
    freq:
        Pandas-compatible frequency string (default ``"H"`` = hourly).
        Examples: ``"D"`` (daily), ``"15T"`` (15-minute), ``"M"`` (monthly).

    Returns
    -------
    Dict with ``"status"``: ``"SUCCESS"``, ``"FAILED"``, or ``"ERROR"``.
    Never raises for expected failure paths — callers match on ``status``.
    """
    task_id: str = self.request.id
    progress = _ProgressReporter(task=self, task_id=task_id)

    logger.info(
        "Forecasting job started",
        extra={
            "task_id": task_id,
            "dataset_path": dataset_path,
            "time_column": time_column,
            "target_column": target_column,
            "freq": freq,
        },
    )

    try:
        
        progress.report("ingesting_temporal_data", percent=5)
        lazy_df = pl.scan_csv(dataset_path, try_parse_dates=True)

        #  Temporal alignment + gap detection (inside pipeline.run())
        progress.report("validating_time_continuity", percent=20)
        pipeline = TemporalPipeline(
            lazy_frame=lazy_df,
            datetime_column=time_column,
            target_column=target_column,
            freq=freq,
            context={"task_id": task_id},
        )

        #  rolling  seasonality feature generation
        progress.report("generating_lags_and_seasonal_features", percent=40)

        #  Walk-forward cross-validation (backtesting)
        progress.report("walk_forward_backtesting", percent=70)

        result = pipeline.run()

        # Emit success metric + return
        progress.report("finalising", percent=90)
        track_training_latency(task_id=task_id, status="success", pipeline="forecasting")

        logger.info(
            "Forecasting model trained successfully",
            extra={
                "task_id": task_id,
                "model_name": result.model_name,
                "horizon": result.horizon,
                "mae": result.metrics.get("mae"),
                "rmse": result.metrics.get("rmse"),
            },
        )
        return _success(task_id, result, freq)

    except FileNotFoundError:
        # Missing file — retrying won't help.
        logger.error(
            "Dataset not found",
            extra={"task_id": task_id, "dataset_path": dataset_path},
        )
        return _failure(task_id, f"Dataset not found: {dataset_path!r}")

    except MemoryError:
        # Dataset too large for this worker — retrying won't help.
        logger.error(
            "Worker out of memory during forecasting",
            extra={"task_id": task_id, "dataset_path": dataset_path},
        )
        return _failure(
            task_id,
            "Worker out of memory — consider chunking the dataset or scaling the worker.",
        )

    except MaxRetriesExceededError:
        logger.error(
            "Forecasting task exceeded max retries",
            extra={"task_id": task_id, "dataset_path": dataset_path},
        )
        return _failure(task_id, "Max retries exceeded — check storage connectivity.")

    except Exception as e:
        
        logger.exception(
            "Unexpected error during forecasting",
            extra={"task_id": task_id, "dataset_path": dataset_path}
        )
        track_training_latency(task_id=task_id, status="failed", pipeline="forecasting")
        
        return _error(task_id, str(e))