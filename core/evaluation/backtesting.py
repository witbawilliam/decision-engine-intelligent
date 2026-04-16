from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any, Protocol, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import polars as pl
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


logger = logging.getLogger(__name__)


class ModelInterface(Protocol):
    def fit(self, df: pl.DataFrame) -> None: ...
    def predict(self, horizon: int) -> Any: ...

@dataclass(frozen=True)
class BacktestResult:
    """Immutable record of backtest performance."""
    strategy: str
    fold_metrics: List[Dict[str, float]]
    avg_mae: float
    avg_rmse: float
    volatility_mae: float  
    execution_time_sec: float

@dataclass
class RegimeComparison:
    """ comparison of Expanding vs Sliding behavior."""
    winning_strategy: str
    drift_detected: bool
    performance_gap: float

class TimeSeriesBacktester:
    

    def __init__(
        self,
        df: pl.DataFrame,
        datetime_column: str,
        target_column: str,
        forecast_horizon: int,
        n_jobs: int = -1  
    ):
        self.df = df.sort(datetime_column)
        self.datetime_column = datetime_column
        self.target_column = target_column
        self.horizon = forecast_horizon
        self.n_jobs = n_jobs

    

    def run_backtest(
        self,
        strategy: str,
        model_factory: Callable[[], ModelInterface],
        initial_train_size: Optional[int] = None,
        window_size: Optional[int] = None,
        step: int = 1,
    ) -> BacktestResult:
        """
        Unified API for both strategies with performance logging.
        """
        start_ts = time.perf_counter()
        folds = self._generate_folds(strategy, initial_train_size, window_size, step)
        
        metrics = []
        
        for train, test in folds:
            fold_metric = self._evaluate_fold(model_factory, train, test)
            metrics.append(fold_metric)

        duration = time.perf_counter() - start_ts
        return self._aggregate(strategy, metrics, duration)

    def _generate_folds(self, strategy: str, initial_size: int, window_size: int, step: int):
        
        if strategy == "expanding":
            for end in range(initial_size, self.df.height - self.horizon, step):
                yield self.df[:end], self.df[end : end + self.horizon]
        
        elif strategy == "sliding":
            for start in range(0, self.df.height - window_size - self.horizon, step):
                yield (
                    self.df[start : start + window_size],
                    self.df[start + window_size : start + window_size + self.horizon]
                )

    def _evaluate_fold(self, model_factory, train: pl.DataFrame, test: pl.DataFrame) -> Dict[str, float]:
        
        model = model_factory()
        model.fit(train)
        
        forecast = model.predict(self.horizon)
        
        if isinstance(forecast, dict) and "yhat" in forecast:
            y_pred = forecast["yhat"][-self.horizon:]
        elif hasattr(forecast, "forecast"): 
            y_pred = forecast.forecast["yhat"][-self.horizon:]
        else:
            y_pred = forecast

        y_true = test[self.target_column].to_numpy()
        
        return self._compute_metrics(y_true, np.array(y_pred))



    def analyze_regime_change(
        self, 
        expanding_res: BacktestResult, 
        sliding_res: BacktestResult
    ) -> RegimeComparison:
        """
        Detects if the environment is changing (Concept Drift).
        Logic: If Sliding Window significantly outperforms Expanding, 
        old data is likely 'poisoning' the model with obsolete patterns.
        """
        gap = (expanding_res.avg_mae - sliding_res.avg_mae) / expanding_res.avg_mae
        
        drift = gap > 0.15
        winner = "sliding" if sliding_res.avg_mae < expanding_res.avg_mae else "expanding"
        
        logger.warning(f"Regime Analysis: Winner={winner}, Drift={drift}, Gap={gap:.2%}")
        
        return RegimeComparison(
            winning_strategy=winner,
            drift_detected=drift,
            performance_gap=gap
        )

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred)))
        }

    def _aggregate(self, strategy: str, metrics: List[Dict[str, float]], duration: float) -> BacktestResult:
        maes = [m["mae"] for m in metrics]
        rmses = [m["rmse"] for m in metrics]
        
        return BacktestResult(
            strategy=strategy,
            fold_metrics=metrics,
            avg_mae=float(np.mean(maes)),
            avg_rmse=float(np.mean(rmses)),
            volatility_mae=float(np.std(maes)), 
            execution_time_sec=duration
        )