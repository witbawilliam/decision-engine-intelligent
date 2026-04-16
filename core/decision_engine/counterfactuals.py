from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Tuple, Optional, Protocol

import polars as pl
from pydantic import BaseModel, Field, ConfigDict
from .uptimization_strategy import BinaryRefinementSearch
from .manifold_guard import ManifoldGuard
from core.contracts.schema_validator import SchemaValidator, BaseModel, ColumnContract


NUMERIC_DTYPES = {
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                pl.Float32, pl.Float64,
        }




class PredictModel(Protocol):
    def predict(self, data: pl.DataFrame) -> Any:
        ...



class OptimizationStatus(str, Enum):
    SAFE = "SAFE"
    MODERATE_RISK = "MODERATE_RISK"
    HIGH_RISK = "HIGH_RISK"
    UNREACHABLE = "UNREACHABLE"
    FAILED = "FAILED"




class CounterfactualResult(BaseModel):
    """
    Immutable API-safe counterfactual response.
    """

    model_config = ConfigDict(frozen=True)

    lever_column: str
    original_value: float
    optimized_value: float
    target_goal: float
    achieved_prediction: float
    risk_score: float = Field(..., ge=0.0, le=1.0)
    status: OptimizationStatus
    message: Optional[str] = None


class CounterfactualOrchestrator:


    def __init__(
        self,
        model: PredictModel,
        training_data: pl.DataFrame,
        risk_threshold: float = 0.7,
        reachability_tolerance: float = 0.05,
    ):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._model = model

        if not hasattr(model, "predict"):
            raise TypeError("Model must implement a 'predict(pl.DataFrame)' method.")

        if training_data.is_empty():
            raise ValueError("Guard initialization failed: Training data manifold is empty.")

        self.risk_threshold = float(risk_threshold)
        self.tolerance = float(reachability_tolerance)

        
        self._optimizer = BinaryRefinementSearch(model)
        self._guard = ManifoldGuard()
        self._guard.fit(training_data)

    
    

    def explain_how_to_hit_target(
        self,
        original_row: pl.DataFrame,
        target_goal: float,
        lever_col: str,
        bounds: Tuple[float, float],
        strict: bool = False,
    ) -> CounterfactualResult:

        if strict:
            return self._execute_core_logic(original_row, target_goal, lever_col, bounds)

        try:
            return self._execute_core_logic(original_row, target_goal, lever_col, bounds)
        except Exception as e:
            self._logger.exception("Counterfactual safe-mode failure")
            return self._build_failure_response(
                original_row,
                lever_col,
                target_goal,
                str(e),
            )

    
    

    def _execute_core_logic(
        self,
        original_row: pl.DataFrame,
        target_goal: float,
        lever_col: str,
        bounds: Tuple[float, float],
    ) -> CounterfactualResult:

        self._validate_inputs(original_row, lever_col, bounds)

        optimized_val = float(
            self._optimizer.optimize(
                original_row,
                lever_col,
                float(target_goal),
                bounds,
            )
        )

        proposed_row = original_row.with_columns(
            pl.lit(optimized_val).alias(lever_col)
        )

        prediction_raw = self._model.predict(proposed_row)

        if not hasattr(prediction_raw, "__len__") or len(prediction_raw) == 0:
            raise RuntimeError("Model returned empty prediction.")

        final_prediction = float(prediction_raw[0])

        risk_score = float(self._guard.get_risk_score(proposed_row))

        status = self._evaluate_status(
            final_prediction,
            float(target_goal),
            risk_score,
        )

        return CounterfactualResult(
            lever_column=lever_col,
            original_value=float(original_row.get_column(lever_col)[0]),
            optimized_value=optimized_val,
            target_goal=float(target_goal),
            achieved_prediction=final_prediction,
            risk_score=min(max(risk_score, 0.0), 1.0),
            status=status,
            message=self._build_message(status),
        )

    
    
    

    def _evaluate_status(
        self,
        prediction: float,
        goal: float,
        risk: float,
    ) -> OptimizationStatus:

        deviation = abs(prediction - goal) / (abs(goal) + 1e-9)

        if deviation > self.tolerance:
            return OptimizationStatus.UNREACHABLE

        if risk > self.risk_threshold:
            return OptimizationStatus.HIGH_RISK

        if risk > (self.risk_threshold / 2):
            return OptimizationStatus.MODERATE_RISK

        return OptimizationStatus.SAFE

    def _build_message(self, status: OptimizationStatus) -> str:
        if status == OptimizationStatus.SAFE:
            return "Target achieved within safe manifold."
        if status == OptimizationStatus.MODERATE_RISK:
            return "Target achieved but moderate extrapolation risk."
        if status == OptimizationStatus.HIGH_RISK:
            return "Target achieved but high extrapolation risk."
        if status == OptimizationStatus.UNREACHABLE:
            return "Target unreachable within given bounds."
        return "Optimization failed."

    

    def _validate_inputs(
        self,
        row: pl.DataFrame,
        col: str,
        bnds: Tuple[float, float],
    ):

        if row.height != 1:
            raise ValueError("Context must be exactly one row (1xN).")

        if col not in row.columns:
            raise KeyError(f"Feature '{col}' missing from context row.")

        if  row.schema[col] not in NUMERIC_DTYPES:

            raise TypeError(f"Lever column '{col}' must be numeric.")

        if bnds[0] >= bnds[1]:
            raise ValueError("Lower bound must be strictly less than upper bound.")

    

    def _build_failure_response(
        self,
        row: pl.DataFrame,
        col: str,
        goal: float,
        error_msg: str,
    ) -> CounterfactualResult:

        original_val = 0.0
        if col in row.columns and row.height == 1:
            try:
                original_val = float(row.get_column(col)[0])
            except Exception:
                pass

        return CounterfactualResult(
            lever_column=col,
            original_value=original_val,
            optimized_value=original_val,
            target_goal=float(goal),
            achieved_prediction=0.0,
            risk_score=1.0,
            status=OptimizationStatus.FAILED,
            message=error_msg,
        )
