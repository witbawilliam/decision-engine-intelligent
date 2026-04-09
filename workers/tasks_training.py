
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
  
    return pl.scan_csv(dataset_path)


def _build_success_result(task_id: str, result: PipelineResult) -> TaskResult:
    """
    Bridges the PipelineResult to V1.JobResultResponse.
    """
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
    name="workers.tasks_training.tabular_task",
    queue="training",
    time_limit=3600,          
    soft_time_limit=3540,     
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

    with RequestContext(trace_id=task_id):
        logger.info("Training job started", extra={"dataset_path": dataset_path})

        try:
            progress.report("loading_data", percent=10)
            lazy_df = _load_data(dataset_path)

            progress.report("training_model", percent=30)
            
            pipeline = TabularPipeline(
                dataframe=lazy_df, 
                target_column=target_column,
                
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
    name="workers.tasks_training.temporal_task",
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

        
        config = TemporalPipelineConfig(**(config_dict or {}))

        progress.report("feature_engineering", percent=30)
        pipeline = TemporalPipeline(
            dataframe=df,
            target_column=target_column,
            datetime_column=datetime_column,
            experiment_id=task_id,
            config=config
        )

        progress.report("training_model", percent=60)
        result = pipeline.run() 

        progress.report("finalising", percent=90)
        return _build_success_result(task_id, result)

    except Exception as e:
        logger.exception("Temporal pipeline failed", extra={"task_id": task_id})
        return _build_error_result(task_id, str(e))