from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Union, Any

import numpy as np

# Structured logging for production traceability
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class RegressionMetricResult:
    """Immutable diagnostic container for regression performance."""
    mae: float
    mse: float
    rmse: float
    r2: Optional[float]
    adjusted_r2: Optional[float]
    mape: Optional[float]
    median_absolute_error: float
    explained_variance: Optional[float]
    max_error: float            # Added: Worst-case scenario
    mean_squared_log_error: Optional[float]  # Added: For exponential growth targets

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

class RegressionMetrics:
    """
    High-Performance Regression Evaluation Engine.
    Engineered for numerical stability and high-dimensional feature spaces.
    """

    def __init__(self, epsilon: float = 1e-9):
        self.epsilon = epsilon
        

    def evaluate(
        self,
        y_true: Union[np.ndarray, list],
        y_pred: Union[np.ndarray, list],
        n_features: Optional[int] = None
    ) -> RegressionMetricResult:
        """
        Calculates a full suite of regression diagnostics with safety guards.
        """
        # Convert to flat numpy arrays for vectorized performance
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        if y_true.size == 0:
            raise ValueError("Input arrays cannot be empty.")
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}")

        # Core error metrics
        errors = y_true - y_pred
        abs_errors = np.abs(errors)
        
        mae = float(np.mean(abs_errors))
        mse = float(np.mean(errors ** 2))
        rmse = np.sqrt(mse)
        median_abs = float(np.median(abs_errors))
        max_err = float(np.max(abs_errors))

        # Statistical fitness metrics
        r2 = self._r2(y_true, y_pred)
        adj_r2 = self._adjusted_r2(r2, len(y_true), n_features)
        mape = self._mape(y_true, y_pred)
        explained_var = self._explained_variance(y_true, y_pred)
        msle = self._msle(y_true, y_pred)

        return RegressionMetricResult(
            mae=mae,
            mse=mse,
            rmse=rmse,
            r2=r2,
            adjusted_r2=adj_r2,
            mape=mape,
            median_absolute_error=median_abs,
            explained_variance=explained_var,
            max_error=max_err,
            mean_squared_log_error=msle
        )

    
    # STABLE METRIC DEFINITIONS
    

    def _r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
        """Calculates R-Squared with zero-variance protection."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot < self.epsilon:
            logger.warning("R2 undefined: Target variable has zero variance.")
            return None
        return float(1 - (ss_res / ss_tot))

    def _adjusted_r2(self, r2: Optional[float], n: int, p: Optional[int]) -> Optional[float]:
        """Adjusts R2 for the number of predictors to prevent overfitting artifacts."""
        if r2 is None or p is None:
            return None
        if n <= p + 1:
            logger.warning("Adjusted R2 undefined: n_samples <= n_features + 1.")
            return None
        return float(1 - (1 - r2) * (n - 1) / (n - p - 1))

    def _mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
        """Vectorized MAPE with divide-by-zero protection."""
        mask = np.abs(y_true) > self.epsilon
        if not np.any(mask):
            return None
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    def _msle(self, y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
        """Mean Squared Logarithmic Error: Useful for targets with exponential growth."""
        if np.any(y_true < 0) or np.any(y_pred < 0):
            return None # MSLE is undefined for negative values
        return float(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))

    def _explained_variance(self, y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
        """Measures how much of the variance in the target the model captures."""
        y_diff = y_true - y_pred
        var_res = np.var(y_diff)
        var_true = np.var(y_true)
        if var_true < self.epsilon:
            return None
        return float(1 - (var_res / var_true))