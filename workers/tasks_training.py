
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional
from datetime import datetime, timezone

import polars as pl
from celery import Task, shared_task
from celery.exceptions import MaxRetriesExceededError

from core.pipelines.tabular_pipeline import TabularPipeline, PipelineResult
from core.pipelines.temporal_pipeline import TemporalPipeline, TemporalPipelineConfig
from monitoring.logging_config import get_logger, RequestContext
from app.schemas.job_schema import V1


logger = get_logger(__name__)


TaskResult = dict[str, Any]

STATUS_SUCCESS = "SUCCESS"
STATUS_FAILED  = "FAILED"
STATUS_ERROR   = "ERROR"



@dataclass
class TaskProgressReporter:
    """
    Thin wrapper around ``self.update_state`` that standardises the progress
    payload shape and keeps progress-reporting calls out of business logic.

    Usage
    -----
    progress = TaskProgressReporter(task=self, task_id=task_id)
    progress.report("loading_data", percent=10)
    """

    task: Task
    task_id: str

    def report(self, step: str, percent: int) -> None:
        self.task.update_state(
            state="PROGRESS",
            meta={"step": step, "percent": percent, "task_id": self.task_id},
        )
        logger.debug(
            "Task progress",
            extra={"task_id": self.task_id, "step": step, "percent": percent},
        )




def _load_data(dataset_path: str) -> pl.LazyFrame:
    """
    Return a *lazy* Polars frame for ``dataset_path``.

    Keeping it lazy lets ``TabularPipeline`` push predicates and projections
    down to the CSV reader, avoiding loading columns / rows it doesn't need.

    Raises
    ------
    FileNotFoundError
        Re-raised from Polars so the task decorator can handle it correctly.
    OSError
        Propagated for the autoretry machinery to catch.
    """
    return pl.scan_csv(dataset_path)


def _build_success_result(task_id: str, result: PipelineResult) -> TaskResult:
    """
    Bridges the PipelineResult to V1.JobResultResponse.
    """
    # Validate and construct the schema
    response = V1.JobResultResponse(
        job_id=task_id,
        status="completed",
        model_version=result.model_version,
        performance=result.metrics,
        insight_summary=f"Successfully trained {result.model_name}"
    )

    # Return as dict with internal status for Celery logic
    payload = response.model_dump()
    payload["status"] = STATUS_SUCCESS
    payload["model_name"] = result.model_name      
    payload["artifacts_path"] = result.artifacts_path 
    return payload


def _build_failuif_result(task_id: str, message: str) -> TaskResult:
    """
    Constructs a standardized error payload.
    In a real V1 schema, you might eventually create a V1.ErrorResponse,
    but for now, we maintain the contract.
    """
    return {
        "status": STATUS_ERROR,
        "task_id": task_id,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


def _build_error_result(task_id: str, message: str) -> TaskResult:
    """
    Constructs a standardized error payload.
    In a real V1 schema, you might eventually create a V1.ErrorResponse,
    but for now, we maintain the contract.
    """
    return {
        "status": STATUS_ERROR,
        "task_id": task_id,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }



@dataclass
class TaskProgressReporter:
    task: Task
    task_id: str

    def report(self, step: str, percent: int) -> None:
        # Connect to V1 Schema for validation
        payload = V1.JobStatusResponse(
            job_id=self.task_id,
            status="running",
            progress=percent,
            updated_at=datetime.now(timezone.utc)
        )
        
        self.task.update_state(
            state="PROGRESS",
            meta=payload.model_dump()
        )
        logger.debug("Task progress reported", extra=payload.model_dump())



class TrainingTask(Task):
    """
    Celery base class for training jobs.

    Responsibilities
    ----------------
    * Structured failure logging (no PII, no large payloads).
    * Hook point for future telemetry (Prometheus counter, Sentry breadcrumb).
    """

    abstract = True  

    def on_failure(
        self,
        exc: Exception,
        task_id: str,
        args: tuple,
        kwargs: dict,
        einfo: Any,
    ) -> None:
        
        logger.error(
            "Training task permanently failed",
            extra={
                "task_id": task_id,
                "exc_type": type(exc).__name__,
                "exc_message": str(exc),
                # args[0] is dataset_path — safe to log, no PII.
                "dataset_path": args[0] if args else kwargs.get("dataset_path"),
            },
            exc_info=einfo,
        )

    def on_retry(
        self,
        exc: Exception,
        task_id: str,
        args: tuple,
        kwargs: dict,
        einfo: Any,
    ) -> None:
        logger.warning(
            "Training task retrying",
            extra={
                "task_id": task_id,
                "exc_type": type(exc).__name__,
                "attempt": self.request.retries + 1,
                "max_retries": self.max_retries,
            },
        )




@shared_task(
    bind=True,
    base=TrainingTask,
    name="workers.tasks.train_tabular_task",
    queue="training",
    time_limit=3600,          
    soft_time_limit=3540,     
    # Only retry infrastructure / I/O errors — logic bugs won't self-heal.
    autoretry_for=(OSError, IOError),
    dont_autoretry_for=(FileNotFoundError,),
    max_retries=3,
    retry_backoff=True,       
    retry_backoff_max=120,
    retry_jitter=True,        
)

def train_tabular_task(self, dataset_path: str, target_column: str) -> TaskResult:
    task_id = self.request.id
    progress = TaskProgressReporter(task=self, task_id=task_id)

    # Wrap EVERYTHING in the RequestContext so all internal logs share the trace_id
    with RequestContext(trace_id=task_id):
        logger.info("Training job started", extra={"dataset_path": dataset_path})

        try:
            progress.report("loading_data", percent=10)
            lazy_df = _load_data(dataset_path)

            progress.report("training_model", percent=30)
            
            pipeline = TabularPipeline(
                dataframe=lazy_df, # Changed from lazy_frame to dataframe
                target_column=target_column,
                # if your TabularPipeline doesn't take task_context, remove it!
            )

            result = pipeline.run()

            progress.report("finalising", percent=90)
            return _build_success_result(task_id, result)

        except Exception as e:
            logger.exception("Unexpected error during training")
            return _build_error_result(task_id, str(e))
        

@shared_task(
    bind=True,
    base=TrainingTask,
    name="workers.tasks.train_temporal_task",
    queue="training",
    time_limit=3600,
    soft_time_limit=3540,
    autoretry_for=(OSError, IOError),
    max_retries=3,
    retry_backoff=True,
)
def train_temporal_task(
    self,
    dataset_path: str,
    target_column: str,
    datetime_column: str,  # Extra arg required for Temporal
    config_dict: Optional[Dict[str, Any]] = None,
) -> TaskResult:
    """
    Train a forecasting model (Prophet) on temporal data.
    """
    task_id: str = self.request.id
    progress = TaskProgressReporter(task=self, task_id=task_id)

    logger.info(
        "Temporal training job started",
        extra={
            "task_id": task_id, 
            "dataset_path": dataset_path, 
            "datetime_column": datetime_column
        },
    )

    try:
    
        progress.report("loading_data", percent=10)
        df = _load_data(dataset_path).collect() 

        # 2. Setup Configuration
        config = TemporalPipelineConfig(**(config_dict or {}))

        # 3. Build and Run Pipeline
        progress.report("feature_engineering", percent=30)
        pipeline = TemporalPipeline(
            dataframe=df,
            target_column=target_column,
            datetime_column=datetime_column,
            experiment_id=task_id,
            config=config
        )

        progress.report("training_model", percent=60)
        # Note: In your class, this calls _train which triggers _register_model
        result = pipeline.run() 

        progress.report("finalising", percent=90)
        return _build_success_result(task_id, result)

    except Exception as e:
        # Standard error handling from your existing boilerplate
        logger.exception("Temporal pipeline failed", extra={"task_id": task_id})
        return _build_error_result(task_id, str(e))