from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Union, Any

import numpy as np


logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class RegressionMetricResult:
    
    mae: float
    mse: float
    rmse: float
    r2: Optional[float]
    adjusted_r2: Optional[float]
    mape: Optional[float]
    median_absolute_error: float
    explained_variance: Optional[float]
    max_error: float            
    mean_squared_log_error: Optional[float]  

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

class RegressionMetrics:
    

    def __init__(self, epsilon: float = 1e-9):
        self.epsilon = epsilon
        

    def evaluate(
        self,
        y_true: Union[np.ndarray, list],
        y_pred: Union[np.ndarray, list],
        n_features: Optional[int] = None
    ) -> RegressionMetricResult:
        
        
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
         raise ValueError("Input arrays contain NaN values. Clean your data before evaluation.")
        if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
          raise ValueError("Input arrays contain Inf values.")
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}")

        
        errors = y_true - y_pred
        abs_errors = np.abs(errors)
        
        mae = float(np.mean(abs_errors))
        mse = float(np.mean(errors ** 2))
        rmse = float(np.sqrt(mse))
        median_abs = float(np.median(abs_errors))
        max_err = float(np.max(abs_errors))

        
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

    
    
    

    def _r2(self, y_true, y_pred) -> float:
        y_true  = np.asarray(y_true, dtype=np.float64)
        y_pred  = np.asarray(y_pred, dtype=np.float64)
        ss_res  = np.sum((y_true - y_pred) ** 2)          
        ss_tot  = np.sum((y_true - np.mean(y_true)) ** 2) 
        if ss_tot == 0:

            return 1.0 if ss_res == 0 else 0.0
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
        # Step 1: define mask first
        mask = np.abs(y_true) > self.epsilon

        # Step 2: count EXCLUDED samples (where mask is False)
        n_excluded = int(np.sum(~mask))
        if n_excluded > 0:
            logger.warning(
                f"MAPE: {n_excluded}/{len(y_true)} samples excluded "
                "due to near-zero y_true values. Result may be unreliable."
            )

        # Step 3: if nothing survives the mask, MAPE is undefined
        if not np.any(mask):
            return None

        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    

    def _msle(self, y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
        """Mean Squared Logarithmic Error: Useful for targets with exponential growth."""
        if np.any(y_true < 0):
            return None  
        if np.any(y_pred < 0):
            logger.warning("MSLE: Negative predictions clamped to 0.")
            y_pred = np.maximum(y_pred, 0)
        return float(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))

    def _explained_variance(self, y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
        """Measures how much of the variance in the target the model captures."""
        y_diff = y_true - y_pred
        var_res = np.var(y_diff)
        var_true = np.var(y_true)
        if var_true < self.epsilon:
            return None
        return float(1 - (var_res / var_true))