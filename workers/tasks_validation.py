from __future__ import annotations
 
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
 
import numpy as np
import polars as pl
from celery import Task, shared_task
 
from core.contracts.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerTriggered
from core.drift.drift_detector import DriftDetector, DriftReport
from feedback.error_logger import ErrorLogger
from core.evaluation.feature_importance import XGBExplainer, ExplanationResult
from app.schemas.job_schema import V1, ALLOWED_TRANSITIONS, TERMINAL_STATUSES
from storage.postgres_client import PostgresClient
from core.drift.statistical_tests import DriftStatistics
 
 

@dataclass
class ValidationResult:
    """
    Immutable snapshot of every check performed during task validation.
    Passed downstream to the training scheduler or returned as an API response.
    """
 

    job_id: str
    validated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
 
    # Gate outcomes  (None = gate was not evaluated)
    schema_valid: bool = False
    circuit_ok: bool = False
    drift_report: Optional[DriftReport] = None
    statistical_report: Optional[Dict[str, Any]] = None
    explanation: Optional[ExplanationResult] = None
    
    passed: bool = False
    failure_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
 
    # DB record ID assigned after persistence
    db_record_id: Optional[str] = None
 
    def to_dict(self) -> Dict[str, Any]:
        """Serialisable snapshot for API responses and audit logs."""
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
    """
    Tune thresholds without touching business logic.
    All values are safe-to-deploy defaults; override per environment.
    """
 
    # Circuit-breaker thresholds
    min_quality_score: float = 0.4
    max_drift_score: float = 0.3
    min_model_performance: float = 0.5
    max_training_time_seconds: int = 600
    max_failure_count: int = 5
 
    # KS drift detector threshold (p-value)
    ks_drift_threshold: float = 0.05
 
   
    monitored_features: List[str] = field(default_factory=list)
 
    # Whether to run SHAP explainability (can be disabled for speed)
    run_explainability: bool = True
 
    # PostgreSQL table names
    validation_table: str = "task_validations"
    drift_table: str = "drift_reports"
 
 

class TaskValidator:
    """
    Orchestrates all validation gates for a single ML job submission.
 
    Gate order (fail-fast):
      1. Schema validation        — Pydantic V1 models from job_schema.py
      2. Status-transition check  — ALLOWED_TRANSITIONS guard
      3. Circuit breaker          — quality / drift / performance / time / failures
      4. Drift detection          — KS p-value per feature (DriftDetector)
      5. Statistical tests        — PSI / KS / KL / JS per flagged feature
      6. Explainability           — SHAP + gain importance (optional)
      7. Persist to PostgreSQL    — audit trail via PostgresClient
    """
 
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
        """
        Execute the full validation pipeline and return a ``ValidationResult``.
 
        Parameters
        ----------
        job_payload      : Raw dict that will be parsed by V1.JobCreate.
        reference_df     : Baseline (training) Polars DataFrame.
        current_df       : Incoming (inference / retrain) Polars DataFrame.
        model            : Fitted XGBModel instance.
        feature_names    : Ordered list of model input feature names.
        quality_score    : Pre-computed data quality score in [0, 1].
        model_metric     : Primary model performance metric (e.g. AUC, R²).
        elapsed_s        : Wall-clock seconds the job has been running.
        failure_count    : Consecutive failures on this job so far.
        current_status   : Current FSM state of the job.
        target_status    : Desired FSM state after validation.
        background_df    : Optional SHAP background dataset.
        """
 
        start = time.perf_counter()
 
        # Derive job_id — prefer idempotency_key, fall back to uuid4
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
 
            
            if self.config.run_explainability:
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
 
        except Exception as exc:  # noqa: BLE001
            result.passed = False
            result.failure_reason = f"UnexpectedError: {type(exc).__name__}: {exc}"
            ErrorLogger.log_error(
                component="TaskValidator.run",
                error=exc,
                context={"job_id": job_id},
                
            )
 
        finally:
            #  Persist audit record (always, even on failure)
            try:
                result = self._persist(result)
            except Exception as persist_exc:  # noqa: BLE001
                result.warnings.append(
                    f"Audit persistence failed: {persist_exc}"
                )
                ErrorLogger.log_error(
                    component="TaskValidator.persist",
                    error=persist_exc,
                    context={"job_id": job_id},
                    
                )
 
        elapsed_ms = (time.perf_counter() - start) * 1000
        result.warnings.append(f"Total validation time: {elapsed_ms:.1f} ms")
        return result
 
   
 
    def _gate_schema(
        self, result: ValidationResult, payload: Dict[str, Any]
    ) -> ValidationResult:
        """Parse and validate the raw job payload against V1.JobCreate."""
        try:
            V1.JobCreate(**payload)
            result.schema_valid = True
        except Exception as exc:
            raise ValidationError(f"Schema validation failed: {exc}") from exc
        return result
 
    def _gate_status_transition(
        self,
        result: ValidationResult,
        current: str,
        target: str,
    ) -> ValidationResult:
        """Enforce the FSM transition table defined in job_schema.py."""
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
 
    def _gate_circuit_breaker(
        self,
        result: ValidationResult,
        quality_score: float,
        model_metric: float,
        elapsed_s: float,
        failure_count: int,
    ) -> ValidationResult:
        """Run all five circuit-breaker checks in the correct triage order."""
        cb = self._circuit_breaker
        
        cb.check_data_quality(quality_score)
 
        cb.check_failure_count(failure_count)
 
        cb.check_training_time(elapsed_s)
 
        if model_metric > 0.0:
            cb.check_model_performance(model_metric)
 
        result.circuit_ok = True
        return result
 
    def _gate_drift(
        self,
        result: ValidationResult,
        reference_df: pl.DataFrame,
        current_df: pl.DataFrame,
    ) -> ValidationResult:
        """
        Run KS-based drift detection across all (or configured) features.
        Populates ``result.drift_report``; also fires the circuit-breaker drift
        gate if the mean drift score exceeds the configured threshold.
        """
        features = self.config.monitored_features or None  # None → auto-detect
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
 
    def _gate_statistical_tests(
        self,
        result: ValidationResult,
        reference_df: pl.DataFrame,
        current_df: pl.DataFrame,
        flagged_features: List[str],
    ) -> ValidationResult:
        """
        Run the full PSI / KS / KL / JS suite on each flagged feature.
        Results are attached to ``result.statistical_report``.
        """
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
            except Exception as exc:  # noqa: BLE001
                result.warnings.append(
                    f"Statistical tests failed for '{col}': {exc}"
                )
                ErrorLogger.log_error(
                    component="TaskValidator.statistical_tests",
                    error=exc,
                    context={"job_id": result.job_id, "feature": col},
                    extra={"error_detail": str(exc)}
                )
 
        result.statistical_report = report
        return result
 
    def _gate_explainability(
        self,
        result: ValidationResult,
        model: Any,
        feature_names: List[str],
        current_df: pl.DataFrame,
        background_df: Optional[pl.DataFrame],
    ) -> ValidationResult:
        """
        Generate SHAP + gain-based feature importances.
        Failure is non-fatal: logged as a warning so validation can still pass.
        """
        try:
            explainer = XGBExplainer(
                model=model,
                feature_names=feature_names,
                background_data=background_df,
            )
            explanation = explainer.explain(current_df[feature_names])
            result.explanation = explanation
        except Exception as exc:  # noqa: BLE001
            result.warnings.append(f"Explainability skipped: {exc}")
            ErrorLogger.log_error(
                component="TaskValidator.explainability",
                error=exc,
                context={"job_id": result.job_id},
                
            )
        return result
 
    
 
    def _persist(self, result: ValidationResult) -> ValidationResult:
        """
        Write the validation outcome to PostgreSQL for audit and monitoring.
 
        """
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
    queue="validation"
)

def validate_task(
    job_payload: Dict[str, Any],
    reference_df: pl.DataFrame,
    current_df: pl.DataFrame,
    model: Any,
    feature_names: List[str],
    quality_score: float,
    model_metric: float,
    elapsed_s: float,
    config: Optional[TaskValidationConfig] = None,
    **kwargs: Any,
  )-> ValidationResult:
    
    return TaskValidator(config=config).run(
        job_payload=job_payload,
        reference_df=reference_df,
        current_df=current_df,
        model=model,
        feature_names=feature_names,
        quality_score=quality_score,
        model_metric=model_metric,
        elapsed_s=elapsed_s,
        **kwargs,
    )
