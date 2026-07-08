from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import os
import tempfile

import numpy as np
import polars as pl
from celery import Task, shared_task

from core.contracts.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerTriggered
from core.drift.drift_detector import DriftDetector, DriftReport
from feedback.error_logger import ErrorLogger
from core.evaluation.feature_importance import XGBExplainer, ExplanationResult
from app.schemas.job_schema import V1, ALLOWED_TRANSITIONS, TERMINAL_STATUSES
from storage.postgres_client import PostgresClient
from storage.s3_client import S3Client          
from core.drift.statistical_tests import DriftStatistics


@dataclass
class ValidationResult:
    job_id: str
    validated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    schema_valid: bool = False
    circuit_ok: bool = False
    drift_report: Optional[DriftReport] = None
    statistical_report: Optional[Dict[str, Any]] = None
    explanation: Optional[ExplanationResult] = None

    passed: bool = False
    failure_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    db_record_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "validated_at": self.validated_at.isoformat(),
            "schema_valid": self.schema_valid,
            "circuit_ok": self.circuit_ok,
            "drift_detected": self.drift_report.is_drifted if self.drift_report else None,
            "flagged_features": self.drift_report.flagged_features if self.drift_report else [],
            "statistical_report": self.statistical_report,
            "shap_importance": (
                self.explanation.shap_importance if self.explanation else None
            ),
            "gain_importance": (
                self.explanation.gain_importance if self.explanation else None
            ),
            "passed": self.passed,
            "failure_reason": self.failure_reason,
            "warnings": self.warnings,
            "db_record_id": self.db_record_id,
        }


@dataclass
class TaskValidationConfig:
    min_quality_score: float = 0.4
    max_drift_score: float = 0.3
    min_model_performance: float = 0.5
    max_training_time_seconds: int = 600
    max_failure_count: int = 5

    ks_drift_threshold: float = 0.05
    monitored_features: List[str] = field(default_factory=list)
    run_explainability: bool = True

    validation_table: str = "task_validations"
    drift_table: str = "drift_reports"


class TaskValidator:
   

    def __init__(self, config: Optional[TaskValidationConfig] = None):
        self.config = config or TaskValidationConfig()

        self._circuit_breaker = CircuitBreaker(
            CircuitBreakerConfig(
                min_quality_score=self.config.min_quality_score,
                max_drift_score=self.config.max_drift_score,
                min_model_performance=self.config.min_model_performance,
                max_training_time_seconds=self.config.max_training_time_seconds,
                max_failure_count=self.config.max_failure_count,
            )
        )
        self._drift_detector = DriftDetector(threshold=self.config.ks_drift_threshold)

    def run(
        self,
        *,
        job_payload: Dict[str, Any],
        reference_df: pl.DataFrame,
        current_df: pl.DataFrame,
        model: Any,
        feature_names: List[str],
        quality_score: float,
        model_metric: float,
        elapsed_s: float,
        failure_count: int = 0,
        current_status: str = "uploaded",
        target_status: str = "validated",
        background_df: Optional[pl.DataFrame] = None,
    ) -> ValidationResult:

        start = time.perf_counter()
        job_id: str = job_payload.get("idempotency_key") or str(uuid.uuid4())
        result = ValidationResult(job_id=job_id)

        try:
            result = self._gate_schema(result, job_payload)
            result = self._gate_status_transition(result, current_status, target_status)
            result = self._gate_circuit_breaker(
                result, quality_score, model_metric, elapsed_s, failure_count
            )
            result = self._gate_drift(result, reference_df, current_df)

            if result.drift_report and result.drift_report.flagged_features:
                result = self._gate_statistical_tests(
                    result, reference_df, current_df,
                    result.drift_report.flagged_features,
                )

            
            if self.config.run_explainability and model is not None:
                result = self._gate_explainability(
                    result, model, feature_names, current_df, background_df
                )

            result.passed = True

        except CircuitBreakerTriggered as exc:
            result.passed = False
            result.failure_reason = f"CircuitBreaker: {exc}"
            ErrorLogger.log_error(
                component="TaskValidator.circuit_breaker",
                error=exc,
                context={"job_id": job_id},
            )

        except ValidationError as exc:
            result.passed = False
            result.failure_reason = f"ValidationError: {exc}"
            ErrorLogger.log_error(
                component="TaskValidator.schema",
                error=exc,
                context={"job_id": job_id},
            )

        except Exception as exc:
            result.passed = False
            result.failure_reason = f"UnexpectedError: {type(exc).__name__}: {exc}"
            ErrorLogger.log_error(
                component="TaskValidator.run",
                error=exc,
                context={"job_id": job_id},
            )

        finally:
            try:
                result = self._persist(result)
            except Exception as persist_exc:
                result.warnings.append(f"Audit persistence failed: {persist_exc}")
                ErrorLogger.log_error(
                    component="TaskValidator.persist",
                    error=persist_exc,
                    context={"job_id": job_id},
                )

        elapsed_ms = (time.perf_counter() - start) * 1000
        result.warnings.append(f"Total validation time: {elapsed_ms:.1f} ms")
        return result

    

    def _gate_schema(self, result, payload):
        try:
            V1.JobCreate(**payload)
            result.schema_valid = True
        except Exception as exc:
            raise ValidationError(f"Schema validation failed: {exc}") from exc
        return result

    def _gate_status_transition(self, result, current, target):
        allowed = ALLOWED_TRANSITIONS.get(current, frozenset())
        if target not in allowed:
            raise ValidationError(
                f"Illegal status transition: '{current}' → '{target}'. "
                f"Allowed: {sorted(allowed) or 'none (terminal state)'}."
            )
        if current in TERMINAL_STATUSES:
            raise ValidationError(
                f"Job is already in terminal status '{current}' and cannot be modified."
            )
        return result

    def _gate_circuit_breaker(self, result, quality_score, model_metric, elapsed_s, failure_count):
        cb = self._circuit_breaker
        cb.check_data_quality(quality_score)
        cb.check_failure_count(failure_count)
        cb.check_training_time(elapsed_s)
        if model_metric > 0.0:
            cb.check_model_performance(model_metric)
        result.circuit_ok = True
        return result

    def _gate_drift(self, result, reference_df, current_df):
        features = self.config.monitored_features or None
        drift_report = self._drift_detector.check_drift(
            reference_df=reference_df,
            current_df=current_df,
            features=features,
        )
        result.drift_report = drift_report

        if drift_report.drift_scores:
            mean_drift = float(np.mean(list(drift_report.drift_scores.values())))
            self._circuit_breaker.check_drift(1.0 - mean_drift)

        if drift_report.is_drifted:
            result.warnings.append(
                f"Drift detected in {len(drift_report.flagged_features)} feature(s): "
                f"{drift_report.flagged_features}"
            )
        return result

    def _gate_statistical_tests(self, result, reference_df, current_df, flagged_features):
        report: Dict[str, Any] = {}
        for col in flagged_features:
            if col not in reference_df.columns or col not in current_df.columns:
                continue
            ref_arr = reference_df[col].drop_nulls().to_numpy()
            cur_arr = current_df[col].drop_nulls().to_numpy()
            if len(ref_arr) == 0 or len(cur_arr) == 0:
                result.warnings.append(
                    f"Skipped statistical tests for '{col}': empty array after null-drop."
                )
                continue
            try:
                report[col] = DriftStatistics.full_report(ref_arr, cur_arr)
            except Exception as exc:
                result.warnings.append(f"Statistical tests failed for '{col}': {exc}")
                ErrorLogger.log_error(
                    component="TaskValidator.statistical_tests",
                    error=exc,
                    context={"job_id": result.job_id, "feature": col},
                    extra={"error_detail": str(exc)}
                )
        result.statistical_report = report
        return result

    def _gate_explainability(self, result, model, feature_names, current_df, background_df):
        try:
            explainer = XGBExplainer(
                model=model,
                feature_names=feature_names,
                background_data=background_df,
            )
            explanation = explainer.explain(current_df[feature_names])
            result.explanation = explanation
        except Exception as exc:
            result.warnings.append(f"Explainability skipped: {exc}")
            ErrorLogger.log_error(
                component="TaskValidator.explainability",
                error=exc,
                context={"job_id": result.job_id},
            )
        return result

    def _persist(self, result: ValidationResult) -> ValidationResult:
        record_id = str(uuid.uuid4())
        validation_row: Dict[str, Any] = {
            "id": record_id,
            "job_id": result.job_id,
            "validated_at": result.validated_at,
            "schema_valid": result.schema_valid,
            "circuit_ok": result.circuit_ok,
            "drift_detected": (
                result.drift_report.is_drifted if result.drift_report else None
            ),
            "flagged_features": (
                result.drift_report.flagged_features if result.drift_report else []
            ),
            "passed": result.passed,
            "failure_reason": result.failure_reason,
            "warnings": result.warnings,
            "shap_importance": (
                result.explanation.shap_importance if result.explanation else None
            ),
        }
        PostgresClient.upsert(
            table=self.config.validation_table,
            data=validation_row,
            conflict_columns=["job_id"],
            returning="id",
        )
        result.db_record_id = record_id

        if result.drift_report and result.drift_report.drift_scores:
            drift_rows = [
                {
                    "id": str(uuid.uuid4()),
                    "validation_id": record_id,
                    "job_id": result.job_id,
                    "feature_name": feature,
                    "ks_p_value": p_value,
                    "is_flagged": feature in result.drift_report.flagged_features,
                    "threshold": result.drift_report.threshold,
                    "recorded_at": result.validated_at,
                }
                for feature, p_value in result.drift_report.drift_scores.items()
            ]
            PostgresClient.bulk_insert(
                table=self.config.drift_table,
                rows=drift_rows,
            )
        return result


class ValidationError(Exception):
    pass




@shared_task(
    bind=True,
    name="workers.tasks_validation.task_validation",
    queue="validation",
)
def validate_task(
    self,                                   
    job_payload: Dict[str, Any],
    config: Optional[Dict] = None,          
) -> Dict[str, Any]:

    job_id = job_payload.get("idempotency_key") or self.request.id

    
    self.update_state(state="STARTED", meta={"progress": 0})
    PostgresClient.upsert(
        table="jobs",
        data={
            "id": job_id,
            "status": "running",
            "updated_at": datetime.now(timezone.utc),
        },
        conflict_columns=["id"],
    )

    try:
        
        s3 = S3Client(
            bucket_name=os.getenv("S3_DATASETS_BUCKET", "ml-datasets"),
            endpoint_url=os.getenv("S3__ENDPOINT_URL"),
            access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region=os.getenv("AWS_REGION", "us-east-1"),
        )

        # download to temp files then read with polars
        with tempfile.TemporaryDirectory() as tmp_dir:
            current_path   = os.path.join(tmp_dir, "current.parquet")
            reference_path = os.path.join(tmp_dir, "reference.parquet")

            s3.download_file(
                object_name=job_payload["s3_key"],
                local_path=current_path,
            )
            s3.download_file(
                object_name=job_payload["reference_s3_key"],
                local_path=reference_path,
            )

            current_df   = pl.read_parquet(current_path)
            reference_df = pl.read_parquet(reference_path)

        task_config = TaskValidationConfig(**(config or {}))

        result = TaskValidator(task_config).run(
            job_payload=job_payload,
            reference_df=reference_df,
            current_df=current_df,
            model=None,                         
            feature_names=list(current_df.columns),
            quality_score=job_payload.get("quality_score", 1.0),
            model_metric=0.0,
            elapsed_s=0.0,
        )

        final_status = "completed" if result.passed else "failed"
        PostgresClient.upsert(
            table="jobs",
            data={
                "id": job_id,
                "status": final_status,
                "progress": 100,
                "updated_at": datetime.now(timezone.utc),
                "failure_reason": result.failure_reason,
            },
            conflict_columns=["id"],
        )

        return result.to_dict()

    except Exception as exc:
        
        PostgresClient.upsert(
            table="jobs",
            data={
                "id": job_id,
                "status": "failed",
                "updated_at": datetime.now(timezone.utc),
                "failure_reason": str(exc),
            },
            conflict_columns=["id"],
        )
        raise