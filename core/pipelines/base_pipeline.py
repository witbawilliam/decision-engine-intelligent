from __future__ import annotations

import logging
import time
import platform
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
from core.contracts.problem_type import ProblemType
import polars as pl





@dataclass(frozen=True)
class PipelineMetadata:
    """Immutable execution audit trail."""
    execution_id: str
    start_time: float
    end_time: float
    duration: float
    step_timings: Dict[str, float]
    system_info: Dict[str, str]


@dataclass(frozen=True)
class PipelineResult:
    """Immutable pipeline output contract."""
    model_name: str
    problem_type: ProblemType
    metrics: Dict[str, float]
    artifacts: Dict[str, Any]
    metadata: PipelineMetadata



# ENTERPRISE BASE PIPELINE


class BasePipeline(ABC):
    """
    Enterprise ML Pipeline Template.

    Guarantees:
    - Deterministic step ordering
    - Telemetry + execution tracing
    - State safety
    - Immutable outputs
    """



    def __init__(
        self,
        dataframe: pl.DataFrame,
        target_column: str,
        datetime_column: Optional[str] = None,
        experiment_id: str = "default_exp",
    ):
        if dataframe is None or dataframe.height == 0:
            raise ValueError("Pipeline requires a non-empty DataFrame.")

        self.df = dataframe
        self.target_column = target_column
        self.datetime_column = datetime_column
        self.experiment_id = experiment_id

        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)

        # Runtime state
        self.problem_type: Optional[ProblemType] = None
        self.model: Any = None
        self.features: List[str] = []
        self.is_fitted: bool = False

        # Diagnostics
        self._step_timings: Dict[str, float] = {}
        self._lifecycle_state: str = "initialized"

    
    

    def run(self) -> PipelineResult:
        """
        Orchestrates full pipeline lifecycle.
        """

        if self.is_fitted:
            raise RuntimeError("Pipeline instance has already been executed.")

        self.logger.info(f"Starting pipeline: {self.experiment_id}")
        global_start = time.perf_counter()

        steps = self._execution_plan()

        try:
            for step_name, step_func in steps:
                self._execute_step(step_name, step_func)

            # Final evaluation isolated
            metrics = self._execute_evaluation()

            artifacts = self._safe_collect_artifacts()

            self.is_fitted = True
            self._lifecycle_state = "completed"

        except Exception as e:
            self._lifecycle_state = "failed"
            self.logger.error(
                f"Pipeline failed during '{step_name}'",
                exc_info=True
            )
            raise e

        global_end = time.perf_counter()

        metadata = PipelineMetadata(
            execution_id=f"{self.experiment_id}_{int(global_start)}",
            start_time=global_start,
            end_time=global_end,
            duration=global_end - global_start,
            step_timings=dict(self._step_timings),
            system_info=self._collect_system_info(),
        )

        return PipelineResult(
            model_name=self.model.__class__.__name__ if self.model else "Unknown",
            problem_type=self.problem_type,
            metrics=metrics,
            artifacts=artifacts,
            metadata=metadata,
        )

    


    def _execution_plan(self):
        """Defines deterministic execution order."""
        return [
            ("validation", self._validate),
            ("problem_detection", self._detect_problem_type),
            ("feature_engineering", self._feature_engineering),
            ("data_splitting", self._split),
            ("model_training", self._train),
        ]

    def _execute_step(self, name: str, func) -> None:
        """Centralized step executor with telemetry."""
        self.logger.info(f"Executing step: {name}")
        start = time.perf_counter()

        func()

        duration = time.perf_counter() - start
        self._step_timings[name] = duration

        self.logger.info(f"Step '{name}' completed in {duration:.4f}s")

    def _execute_evaluation(self) -> Dict[str, float]:
        """Runs evaluation in isolated telemetry context."""
        start = time.perf_counter()

        if self.problem_type is None:
            raise RuntimeError("Problem type was not determined before evaluation.")

        metrics = self._evaluate_final()

        duration = time.perf_counter() - start
        self._step_timings["evaluation"] = duration

        return metrics

    def _safe_collect_artifacts(self) -> Dict[str, Any]:
        """Ensures artifact dictionary is always safe."""
        artifacts = self._collect_artifacts() or {}
        if not isinstance(artifacts, dict):
            raise TypeError("_collect_artifacts must return a dictionary.")
        return artifacts

    def _collect_system_info(self) -> Dict[str, str]:
        """Captures runtime environment metadata."""
        return {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        }

    
    
    

    @abstractmethod
    def _validate(self) -> None:
        pass

    @abstractmethod
    def _detect_problem_type(self) -> None:
        pass

    @abstractmethod
    def _feature_engineering(self) -> None:
        pass

    @abstractmethod
    def _split(self) -> None:
        pass

    @abstractmethod
    def _train(self) -> None:
        pass

    @abstractmethod
    def _evaluate_final(self) -> Dict[str, float]:
        pass

    
    
    

    def _collect_artifacts(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "feature_names": self.features,
        }