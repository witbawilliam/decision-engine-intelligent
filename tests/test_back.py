import pytest
from datetime import datetime, timedelta
from core.evaluation.backtesting import TimeSeriesBacktester, BacktestResult
import numpy as np
import polars as pl

@pytest.fixture
def time_series_data():
    """
    Creates a dataset where the first half has a mean of 10
    and the second half has a mean of 100 (Regime Shift).
    """
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(100)]
    # Regime 1: Low values
    targets = [10.0 + np.random.normal(0, 1) for _ in range(50)]
    # Regime 2: High values
    targets += [100.0 + np.random.normal(0, 1) for _ in range(50)]
    
    return pl.DataFrame({
        "timestamp": dates,
        "target": targets
    })

def test_expanding_window_execution(time_series_data):
    tester = TimeSeriesBacktester(
        df=time_series_data,
        datetime_column="timestamp",
        target_column="target",
        forecast_horizon=5
    )
    
    result = tester.run_backtest(
        strategy="expanding",
        model_factory=model_factory,
        initial_train_size=20,
        step=10
    )
    
    assert isinstance(result, BacktestResult)
    assert len(result.fold_metrics) > 0
    assert result.avg_mae > 0
    assert result.execution_time_sec > 0

def test_regime_change_detection(time_series_data):
    """
    In this test, the Sliding window should outperform the Expanding window
    because the 'Expanding' window is polluted by the old 'Regime 1' (mean 10) 
    data while trying to predict 'Regime 2' (mean 100).
    """
    tester = TimeSeriesBacktester(
        df=time_series_data,
        datetime_column="timestamp",
        target_column="target",
        forecast_horizon=5
    )
    
    # We test on the latter half of the data
    # Sliding window only sees the recent '100' mean data
    # Expanding window still sees the early '10' mean data
    exp_res = tester.run_backtest("expanding", model_factory, initial_train_size=60, step=5)
    sli_res = tester.run_backtest("sliding", model_factory, window_size=10, step=5)
    
    comparison = tester.analyze_regime_change(exp_res, sli_res)
    
    # Assertions
    assert comparison.winning_strategy == "sliding"
    # Because the mean jumped from 10 to 100, the gap will be massive
    assert comparison.drift_detected is True