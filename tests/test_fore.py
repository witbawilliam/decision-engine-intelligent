import pytest
import numpy as np
from core.evaluation.forecasting_metrics import ForecastingMetrics, ForecastMetricResult

@pytest.fixture
def metrics_engine():
    # Initialize with a seasonal period of 3 for testing MASE
    return ForecastingMetrics(seasonal_period=3)

def test_basic_metric_accuracy(metrics_engine):
    """Verify core math for MAE, RMSE, and WAPE."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 35.0]) # Errors: 2, -2, 5
    
    result = metrics_engine.evaluate(y_true, y_pred)
    
    # MAE = (2 + 2 + 5) / 3 = 3.0
    assert pytest.approx(result.mae) == 3.0
    # RMSE = sqrt((4 + 4 + 25) / 3) = sqrt(11) ≈ 3.3166
    assert pytest.approx(result.rmse, rel=1e-3) == 3.3166
    # WAPE = sum(abs_errors) / sum(actuals) = 9 / 60 = 15%
    assert pytest.approx(result.wape) == 15.0

def test_zero_handling_safety(metrics_engine):
    """Enterprise safety: Ensure MAPE doesn't crash on zero actuals."""
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    
    result = metrics_engine.evaluate(y_true, y_pred)
    
    assert result.mape is None  # Should return None instead of Inf
    assert result.smape == 200.0 # sMAPE maxes out at 200% for zero vs non-zero
    assert result.wape == 100.0  # Predicted volume where actual was 0

def test_mase_logic(metrics_engine):
    """
    Verify MASE compares correctly against the training baseline.
    If the model is 2x better than the naive baseline, MASE should be 0.5.
    """
    y_train = np.array([10, 10, 10, 20, 20, 20]) # Seasonal diffs (period 3) = 10
    # Mean Absolute Seasonal Error (naive) = 10.0
    
    y_true = np.array([30, 30, 30])
    y_pred = np.array([25, 25, 25]) # Model Error = 5.0
    
    result = metrics_engine.evaluate(y_true, y_pred, y_train=y_train)
    
    # MASE = Model_Error / Naive_Error = 5 / 10 = 0.5
    assert result.mase == 0.5

def test_bias_direction(metrics_engine):
    """Verify that under-forecasting results in positive bias."""
    y_true = np.array([100, 100])
    y_pred = np.array([80, 80]) # Under-forecasting by 20 each
    
    result = metrics_engine.evaluate(y_true, y_pred)
    
    # (200 - 160) / 200 = 0.20 (20% under-forecast)
    assert result.bias ==pytest.approx (0.20)