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
    bias: float               
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
        y_train: Optional[np.ndarray] = None,
        y_train_groups: Optional[Union[np.ndarray, list]] = None,
    ) -> ForecastMetricResult:
        """
        Entry point for batch evaluation. Performs sanity checks before computation.

        y_train_groups: Optional entity/series identifier aligned 1:1 with y_train
            (e.g. product IDs). When provided, MASE's naive seasonal-lag baseline
            is computed WITHIN each group independently and then pooled — this
            prevents spurious lag differences being computed across the boundary
            between two unrelated series when y_train is a concatenated panel
            (e.g. multiple products stacked into one array). If omitted, y_train
            is treated as a single continuous series, which is only correct for
            genuinely single-series data.
        """

        y_true = np.asarray(y_true).flatten().astype(np.float64)
        y_pred = np.asarray(y_pred).flatten().astype(np.float64)

        valid_mask = (
            np.isfinite(y_true)
            & np.isfinite(y_pred)
        )

        invalid_rows = len(y_true) - np.sum(valid_mask)

        if invalid_rows > 0:
            logger.warning(
                "Dropped %d invalid forecast rows during evaluation.",
                invalid_rows
            )

        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

        if len(y_true) == 0 or len(y_true) != len(y_pred):
            raise ValueError("Input arrays must be non-empty and of equal length.")

        # Core Metrics
        mae = self._mae(y_true, y_pred)
        rmse = self._rmse(y_true, y_pred)
        mape = self._mape(y_true, y_pred)
        smape = self._smape(y_true, y_pred)
        wape = self._wape(y_true, y_pred)
        bias = self._bias(y_true, y_pred)

        # Comparative Metric
        mase = None
        if y_train is not None:
            mase = self._mase(
                y_true,
                y_pred,
                np.asarray(y_train).flatten(),
                groups=np.asarray(y_train_groups).flatten() if y_train_groups is not None else None,
            )

        return ForecastMetricResult(
            mae=mae,
            rmse=rmse,
            mape=mape,
            smape=smape,
            mase=mase,
            wape=wape,
            bias=bias
        )



    def _mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    def _rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def _bias(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
       
        return float(np.sum(y_true - y_pred) / (np.sum(y_true) + 1e-9))

    def _mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
        
        mask = np.abs(y_true) > 1e-9
        if not np.any(mask):
            logger.warning("MAPE undefined: all actuals are zero.")
            return None
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    def _smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        
        denominator = (np.abs(y_true) + np.abs(y_pred))
        mask = denominator > 1e-9
        if not np.any(mask):
            return 0.0

        return float(200.0 * np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]))

    def _wape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        
        actual_sum = np.sum(np.abs(y_true))
        if actual_sum < 1e-9:
            return 0.0 if np.sum(np.abs(y_pred)) < 1e-9 else 100.0
        return float(np.sum(np.abs(y_true - y_pred)) / actual_sum * 100)

    

    def _mase(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        """
        MASE compares model error to a Naive seasonal forecast.
        MASE < 1 means the model is better than just guessing 'yesterday' or 'last season'.

        If `groups` is provided, y_train is treated as a concatenation of
        independent series (e.g. multiple products/entities). The seasonal-lag
        diff (t vs t-m) is computed separately within each group's own
        contiguous slice, then pooled across groups — this avoids computing a
        nonsensical diff across the boundary where one entity's series ends
        and the next one's begins in the flat array.
        """
        m = self.seasonal_period

        if groups is not None:
            if len(groups) != len(y_train):
                raise ValueError(
                    f"y_train_groups length ({len(groups)}) must match y_train length ({len(y_train)})."
                )

            naive_errors = []
            for group_id in np.unique(groups):
                series = y_train[groups == group_id]
                if len(series) <= m:
                    logger.debug(
                        "MASE: group '%s' has %d rows, too short for seasonal_period=%d — skipped.",
                        group_id, len(series), m,
                    )
                    continue
                naive_errors.append(np.abs(series[m:] - series[:-m]))

            if not naive_errors:
                logger.debug("MASE: no group had sufficient history for seasonal lag; MASE undefined.")
                return None

            scale = float(np.mean(np.concatenate(naive_errors)))

        else:
            if len(y_train) <= m:
                logger.debug("MASE: y_train shorter than seasonal period.")
                return None
            naive_error = np.abs(y_train[m:] - y_train[:-m])
            scale = float(np.mean(naive_error))

        if scale < 1e-9:
            return None 

        return float(np.mean(np.abs(y_true - y_pred)) / scale)