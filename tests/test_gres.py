import pytest
import numpy as np
from core.evaluation.regression_metrics import RegressionMetrics, RegressionMetricResult

@pytest.fixture
def engine():
    return RegressionMetrics()

def test_regression_math_accuracy(engine):
    """Verify MAE, RMSE, and R2 with a known simple case."""
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    # Errors: [0.5, -0.5, 0.0, -1.0]
    # Abs Errors: [0.5, 0.5, 0.0, 1.0] -> Mean = 0.5
    # Squared Errors: [0.25, 0.25, 0.0, 1.0] -> Mean = 0.375
    
    result = engine.evaluate(y_true, y_pred)
    
    assert pytest.approx(result.mae) == 0.5
    assert pytest.approx(result.mse) == 0.375
    assert pytest.approx(result.rmse) == np.sqrt(0.375)
    assert result.r2 > 0  # Should be high for this good fit

def test_adjusted_r2_penalty(engine):
    """Verify that Adjusted R2 is lower than R2 as features increase."""
    y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y_pred = y_true + np.random.normal(0, 0.1, 10)
    
    # Evaluate with 1 feature vs 5 features
    res_low_p = engine.evaluate(y_true, y_pred, n_features=1)
    res_high_p = engine.evaluate(y_true, y_pred, n_features=5)
    
    # Adjusted R2 should decrease as we 'pretend' to add more noise features
    assert res_high_p.adjusted_r2 < res_low_p.adjusted_r2
    assert res_high_p.adjusted_r2 < res_high_p.r2

def test_constant_target_stability(engine):
    """Ensure R2 returns None instead of crashing when variance is zero."""
    y_true = np.array([10.0, 10.0, 10.0])
    y_pred = np.array([11.0, 11.0, 11.0])
    
    result = engine.evaluate(y_true, y_pred)
    
    assert result.r2 is None
    assert result.explained_variance is None
    assert result.mae == 1.0

def test_msle_safety(engine):
    """Verify MSLE handles zeros but ignores negative values."""
    y_true = np.array([10, 20, 30])
    y_pred = np.array([11, 19, 31])
    
    # Positive case
    result = engine.evaluate(y_true, y_pred)
    assert result.mean_squared_log_error is not None
    
    # Negative case (MSLE should be None)
    res_neg = engine.evaluate(np.array([-1, 2]), np.array([1, 2]))
    assert res_neg.mean_squared_log_error is None