
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional
from datetime import datetime, timezone
import os
import tempfile
from storage.s3_client import S3Client
from storage.postgres_client import PostgresClient

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





def _load_data(dataset_path: str) -> pl.LazyFrame:
  
    return pl.scan_csv(dataset_path)


def _build_success_result(task_id: str, result: PipelineResult) -> TaskResult:
   
    version = getattr(result, "model_version", "1")

    if "." not in str(version):
        semver_version = f"{version}.0.0" 
    else:
        semver_version = str(version)


    response = V1.JobResultResponse(
        job_id=task_id,
        status="completed",
        model_version=semver_version,
        performance=result.metrics,
        insight_summary=f"Successfully trained {result.model_name}"
    )

    payload = response.model_dump()
    payload["status"] = STATUS_SUCCESS
    payload["model_name"] = result.model_name      
    payload["artifacts_path"] = getattr(result, "artifacts_path", "models/default_path.pkl")
    payload["predictions"] = result.artifacts["predictions"]
    return payload



def _build_error_result(task_id: str, message: str) -> TaskResult:
    
    return {
        "status": STATUS_ERROR,
        "task_id": task_id,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

def _build_s3_client() -> S3Client:

    return S3Client(

        bucket_name=os.getenv("S3_DATASETS_BUCKET", "ml-datasets"),
        endpoint_url=os.getenv("S3__ENDPOINT_URL"),
        access_key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region=os.getenv("AWS_REGION", "us-east-1"),

    )


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
            meta=payload.model_dump(mode="json")
        )
        logger.debug("Task progress reported", extra=payload.model_dump(mode="json"))



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

def train_tabular_task(self, job_payload: dict) -> TaskResult:
    task_id = self.request.id
    job_id  = job_payload.get("idempotency_key") or task_id
    progress = TaskProgressReporter(task=self, task_id=job_id)

    
    PostgresClient.upsert(
        table="jobs",
        data={"id": job_id, "status": "running", "progress": 0,
              "updated_at": datetime.now(timezone.utc)},
        conflict_columns=["id"],
    )

    with RequestContext(trace_id=task_id):
        try:
            
            s3 = _build_s3_client()

            with tempfile.TemporaryDirectory() as tmp_dir:
                local_path = os.path.join(tmp_dir, "dataset.parquet")
                s3.download_file(
                    object_name=job_payload["s3_key"],
                    local_path=local_path,
                )

                progress.report("loading_data", percent=10)
                df = pl.scan_parquet(local_path).collect() 

                progress.report("training_model", percent=30)
                pipeline = TabularPipeline(
                    df=df,
                    target_column=job_payload["target_column"],
                    problem_type=job_payload.get("problem_type"),
                    s3_client=s3
                )

                model_name = job_payload.get("model_name") or job_id
                result = pipeline.execute_pipeline(model_name=model_name)

                



            progress.report("finalising", percent=90)

            
            PostgresClient.upsert(
                table="jobs",
                data={"id": job_id, "status": "completed", "progress": 100,
                      "updated_at": datetime.now(timezone.utc)},
                conflict_columns=["id"],
            )

            return _build_success_result(job_id, result)

        except Exception as e:
            PostgresClient.upsert(
                table="jobs",
                data={"id": job_id, "status": "failed", "progress": 0,
                      "updated_at": datetime.now(timezone.utc),
                      "failure_reason": str(e)},
                conflict_columns=["id"],
            )
            logger.exception("Unexpected error during training")
            raise
        

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
def train_temporal_task(self, job_payload: dict) -> TaskResult:
    task_id = self.request.id
    job_id  = job_payload.get("idempotency_key") or task_id
    progress = TaskProgressReporter(task=self, task_id=job_id)

    PostgresClient.upsert(
        table="jobs",
        data={"id": job_id, "status": "running", "progress": 0,
              "updated_at": datetime.now(timezone.utc)},
        conflict_columns=["id"],
    )

    with RequestContext(trace_id=task_id):
        try:
            s3 = _build_s3_client()  

            with tempfile.TemporaryDirectory() as tmp_dir:
                local_path = os.path.join(tmp_dir, "temporal_dataset.parquet")
                s3.download_file(
                    object_name=job_payload["s3_key"],
                    local_path=local_path,
                )
                progress.report("loading_data", percent=10)
                df = pl.scan_parquet(local_path).collect()

                time_col   = job_payload.get("time_column")
                if time_col is None:
                  time_col = job_payload.get("datetime_column")
  
                target_col = job_payload.get("target_column")

                if not time_col:
                    raise ValueError(f"Missing time_column. Got keys: {list(job_payload.keys())}")
                if not target_col:
                    raise ValueError(f"Missing target_column. Got keys: {list(job_payload.keys())}")

                config = TemporalPipelineConfig(
                    forecast_horizon=job_payload.get("forecast_horizon", 30)
                )
                progress.report("feature_engineering", percent=30)

                pipeline = TemporalPipeline(
                    dataframe=df,
                    target_column=target_col,
                    datetime_column=time_col,
                    experiment_id=job_id,
                    config=config,
                    s3_client=s3
                )
                progress.report("training_model", percent=60)
                result = pipeline.run()
                pipeline.promote_to_production()
                progress.report("finalising", percent=90)

            PostgresClient.upsert(
                table="jobs",
                data={"id": job_id, "status": "completed", "progress": 100,
                      "updated_at": datetime.now(timezone.utc)},
                conflict_columns=["id"],
            )
            return _build_success_result(job_id, result)

        except Exception as e:
            PostgresClient.upsert(
                table="jobs",
                data={"id": job_id, "status": "failed", "progress": 0,
                      "updated_at": datetime.now(timezone.utc),
                      "failure_reason": str(e)},
                conflict_columns=["id"],
            )
            logger.exception("Temporal pipeline failed", extra={"job_id": job_id})
            return _build_error_result(job_id, str(e))



@shared_task(
    bind=True,
    base=TrainingTask,
    name="workers.tasks_training.train_all_products_task",
    queue="training",
    time_limit=7200,
    soft_time_limit=7100,
    autoretry_for=(OSError, IOError),
    max_retries=3,
    retry_backoff=True,
    retry_backoff_max=120,
    retry_jitter=True,
)

def train_all_products_task(self, job_payload: dict) -> TaskResult:
   
    from core.pipelines.temporal_pipeline import train_all_products
 
    task_id  = self.request.id
    job_id   = job_payload.get("idempotency_key") or task_id
    progress = TaskProgressReporter(task=self, task_id=job_id)
 
    # Mark job as running in PostgreSQL
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
 
    with RequestContext(trace_id=task_id):
        try:
            s3 = _build_s3_client()
 
            #  Download dataset from S3 
            with tempfile.TemporaryDirectory() as tmp_dir:
                local_path = os.path.join(tmp_dir, "batch_dataset.parquet")
                s3.download_file(
                    object_name=job_payload["s3_key"],
                    local_path=local_path,
                )
 
                progress.report("loading_data", percent=10)
                df = pl.scan_parquet(local_path).collect()
 
                # Validate required keys 
                product_col     = job_payload.get("product_col")
                target_col      = job_payload.get("target_column")
                datetime_col    = job_payload.get("datetime_column") or job_payload.get("time_column")
 
                if not product_col:
                    raise ValueError(f"Missing product_col. Got keys: {list(job_payload.keys())}")
                if not target_col:
                    raise ValueError(f"Missing target_column. Got keys: {list(job_payload.keys())}")
                if not datetime_col:
                    raise ValueError(f"Missing datetime_column. Got keys: {list(job_payload.keys())}")
 
                config = TemporalPipelineConfig(
                    forecast_horizon=job_payload.get("forecast_horizon", 30)
                )
 
                # Run batch training 
                progress.report("training_products", percent=30)
                results = train_all_products(
                    full_df         = df,
                    product_col     = product_col,
                    target_column   = target_col,
                    datetime_column = datetime_col,
                    s3_client       = s3,
                    config          = config,
                )
                if not results:
                    raise RuntimeError(
                        f"No products met the minimum row threshold for training "
                        f"(product_col={product_col!r}). Zero models trained."
                    )
 
                progress.report("finalising", percent=90)

            products_trained = list(results.keys()) if results else []
            total_trained = len(products_trained)
 
            # Mark completed 
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
 
            return {
                "status":           STATUS_SUCCESS,
                "task_id":          task_id,
                "job_id":           job_id,
                "products_trained": products_trained,
                "total_trained":    total_trained,
                "timestamp":        datetime.now(timezone.utc).isoformat(),
            }
 
        except Exception as e:
            PostgresClient.upsert(
                table="jobs",
                data={
                    "id":             job_id,
                    "status":         "failed",
                    "progress":       0,
                    "updated_at":     datetime.now(timezone.utc),
                    "error_message": str(e),
                },
                conflict_columns=["id"],
            )
            logger.exception(
                "Batch product training failed",
                extra={"job_id": job_id},
            )
            return _build_error_result(job_id, str(e))
 