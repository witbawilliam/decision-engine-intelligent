from __future__ import annotations

import logging
from typing import Dict, Optional, Union
from dataclasses import dataclass, asdict

import numpy as np

# Configure Structured Logging
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ForecastMetricResult:
    """Immutable enterprise-grade evaluation container."""
    mae: float
    rmse: float
    mape: Optional[float]
    smape: float
    mase: Optional[float]
    wape: float
    r2: Optional[float]
    bias: float               # Added: Directional error
    coverage_80: Optional[float] = None # Added: Probabilistic quality

    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in asdict(self).items() if v is not None}

class ForecastingMetrics:
    """
    High-Stability Forecasting Evaluation Engine.
    Handles numerical edge cases (zeros, NaNs) and calculates relative performance.
    """

    def __init__(self, seasonal_period: int = 1):
        """
        seasonal_period: Lag used for MASE (e.g., 7 for daily data with weekly seasonality).
        """
        if seasonal_period < 1:
            raise ValueError("seasonal_period must be at least 1.")
        self.seasonal_period = seasonal_period

    def evaluate(
        self,
        y_true: Union[np.ndarray, list],
        y_pred: Union[np.ndarray, list],
        y_train: Optional[np.ndarray] = None
    ) -> ForecastMetricResult:
        """
        Entry point for batch evaluation. Performs sanity checks before computation.
        """
        # Ensure numpy arrays and handle potential empty inputs
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        if len(y_true) == 0 or len(y_true) != len(y_pred):
            raise ValueError("Input arrays must be non-empty and of equal length.")

        # Core Metrics
        mae = self._mae(y_true, y_pred)
        rmse = self._rmse(y_true, y_pred)
        mape = self._mape(y_true, y_pred)
        smape = self._smape(y_true, y_pred)
        wape = self._wape(y_true, y_pred)
        r2 = self._r2(y_true, y_pred)
        bias = self._bias(y_true, y_pred)

        # Comparative Metric
        mase = None
        if y_train is not None:
            mase = self._mase(y_true, y_pred, np.asarray(y_train).flatten())

        return ForecastMetricResult(
            mae=mae,
            rmse=rmse,
            mape=mape,
            smape=smape,
            mase=mase,
            wape=wape,
            r2=r2,
            bias=bias
        )


    # STABLE METRIC DEFINITIONS


    def _mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    def _rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def _bias(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates normalized forecast bias. 
        Positive = Under-forecasting; Negative = Over-forecasting.
        """
        return float(np.sum(y_true - y_pred) / (np.sum(y_true) + 1e-9))

    def _mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
        """Safe MAPE: Returns None instead of Inf if zeros are present."""
        mask = np.abs(y_true) > 1e-9
        if not np.any(mask):
            logger.warning("MAPE undefined: all actuals are zero.")
            return None
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    def _smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Stable Symmetric MAPE calculation."""
        denominator = (np.abs(y_true) + np.abs(y_pred))
        mask = denominator > 1e-9
        if not np.any(mask):
            return 0.0
        # Formula: 200 * avg(|y - y_hat| / (|y| + |y_hat|))
        return float(200.0 * np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]))

    def _wape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Weighted Absolute Percentage Error (often called Volume Weighted MAPE)."""
        actual_sum = np.sum(np.abs(y_true))
        if actual_sum < 1e-9:
            return 0.0 if np.sum(np.abs(y_pred)) < 1e-9 else 100.0
        return float(np.sum(np.abs(y_true - y_pred)) / actual_sum * 100)

    def _r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot < 1e-9:
            return None # Undefined for constant actuals
        return float(1 - (ss_res / ss_tot))

    def _mase(self, y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> Optional[float]:
        """
        MASE compares model error to a Naive seasonal forecast.
        MASE < 1 means the model is better than just guessing 'yesterday' or 'last season'.
        """
        m = self.seasonal_period
        if len(y_train) <= m:
            logger.debug("MASE: y_train shorter than seasonal period.")
            return None

        # Calculate the absolute scale of a seasonal naive forecast
        naive_error = np.abs(y_train[m:] - y_train[:-m])
        scale = np.mean(naive_error)

        if scale < 1e-9:
            return None # Training data is constant/periodic zero

        return float(np.mean(np.abs(y_true - y_pred)) / scale)