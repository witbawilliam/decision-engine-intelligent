import pytest
import polars as pl
import numpy as np
from typing import Any
from unittest.mock import MagicMock
from core.evaluation.backtesting import TimeSeriesBacktester

class MockModel:
    """Mock model for fast backtesting tests."""
    def fit(self, df: pl.DataFrame) -> None:
        pass  # Training simulation

    def predict(self, horizon: int) -> np.ndarray:
        # Return dummy predictions equal to the horizon length
        return np.array([1.0] * horizon)

def model_factory():
    return MockModel()


class TestTimeSeriesBacktester:
    
    @pytest.fixture
    def sample_data(self):
        """Creates 100 rows of synthetic time series data."""
        return pl.DataFrame({
            "ds": pl.date_range(start=np.datetime64("2026-01-01"), 
                               end=np.datetime64("2026-04-10"), 
                               interval="1d", eager=True),
            "y": np.random.normal(10, 1, 100)
        })

    def test_expanding_window_generation(self, sample_data):
        """Verify that expanding windows grow and don't leak data."""
        tester = TimeSeriesBacktester(sample_data, "ds", "y", forecast_horizon=5)
        
        # Test 1: Initial training size of 20, step of 10
        folds = list(tester._generate_folds("expanding", initial_size=20, window_size=None, step=10))
        
        # Check first fold
        train_1, test_1 = folds[0]
        assert train_1.height == 20
        assert test_1.height == 5
        # Ensure temporal continuity: test starts exactly after train ends
        assert train_1["ds"][-1] < test_1["ds"][0]

        # Check second fold (should have expanded)
        train_2, test_2 = folds[1]
        assert train_2.height == 30 
        assert train_2[:20].equals(train_1)

    def test_sliding_window_generation(self, sample_data):
        """Verify sliding windows maintain a fixed size."""
        tester = TimeSeriesBacktester(sample_data, "ds", "y", forecast_horizon=5)
        window_size = 20
        
        folds = list(tester._generate_folds("sliding", initial_size=None, window_size=window_size, step=10))
        
        train_1, _ = folds[0]
        train_2, _ = folds[1]
        
        assert train_1.height == window_size
        assert train_2.height == window_size
        # In sliding, the start of train_2 should be 10 steps ahead of train_1
        assert train_2["ds"][0] > train_1["ds"][0]

    def test_run_backtest_full_cycle(self, sample_data):
        """Tests the end-to-end run_backtest execution."""
        tester = TimeSeriesBacktester(sample_data, "ds", "y", forecast_horizon=2)
        
        result = tester.run_backtest(
            strategy="expanding",
            model_factory=model_factory,
            initial_train_size=80,
            step=5
        )
        
        assert isinstance(result.avg_mae, float)
        assert len(result.fold_metrics) > 0
        assert result.execution_time_sec > 0
        assert result.strategy == "expanding"

    def test_analyze_regime_change_detection(self):
        """Tests the logic that detects environment shifts."""
        minimal_df = pl.DataFrame({"ds": [], "y": []}) 
        tester = TimeSeriesBacktester(minimal_df, "ds", "y", 5)
        
        # Scenario: Sliding is much better than Expanding (Drift)
        exp_res = MagicMock(avg_mae=10.0)
        slid_res = MagicMock(avg_mae=5.0) # 50% better
        
        comparison = tester.analyze_regime_change(exp_res, slid_res)
        
        assert comparison.drift_detected is True
        assert comparison.winning_strategy == "sliding"
        assert comparison.performance_gap == 0.5