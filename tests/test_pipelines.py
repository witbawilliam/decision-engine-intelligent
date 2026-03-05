import pytest
import numpy as np
import polars as pl
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from core.pipelines.temporal_pipeline import  TemporalPipeline, TemporalPipelineConfig
from core.contracts.circuit_breaker import CircuitBreakerTriggered
# Import your pipeline classes
# from your_module import TemporalPipeline, TemporalPipelineConfig, TemporalResolution

def create_synthetic_temporal_data(n_rows=100):
    """Generates a Polars DataFrame with a trend and seasonality."""
    base_date = datetime(2023, 1, 1)
    df = pl.DataFrame({
        "timestamp": [base_date + timedelta(days=i) for i in range(n_rows)],
        "sales": [100 + i + (10 if i % 7 == 0 else 0) + np.random.normal(0, 2) for i in range(n_rows)],
        "is_promo": np.random.choice([0, 1], n_rows)
    })
    return df

class TestTemporalPipeline:

    @pytest.fixture
    def mock_dependencies(self):
        """Mocks the heavy infrastructure pieces."""
        with patch("core.models.prophet_model.ProphetModel") as mock_prophet, \
             patch("core.models.model_registry.ModelRegistry") as mock_registry, \
             patch("core.contracts.circuit_breaker.CircuitBreaker") as mock_cb:
            
            # Setup Prophet Mock return values
            mock_prophet_inst = mock_prophet.return_value
            mock_prophet_inst.predict.return_value = MagicMock(
                yhat=np.array([110.0]*30),
                forecast_df=pd.DataFrame({"ds": [datetime.now()], "yhat": [110.0], "yhat_lower": [100.0], "yhat_upper": [120.0]})
            )
            
            yield {
                "prophet": mock_prophet,
                "registry": mock_registry,
                "breaker": mock_cb
            }

    def test_pipeline_initialization(self, mock_dependencies):
        df = create_synthetic_temporal_data()
        config = TemporalPipelineConfig(forecast_horizon=14)
        
        pipeline = TemporalPipeline(
            dataframe=df,
            target_column="sales",
            datetime_column="timestamp",
            config=config
        )
        
        assert pipeline.target_column == "sales"
        assert pipeline.config.forecast_horizon == 14
        assert pipeline._lifecycle_state == "initialized"

    def test_full_run_execution(self, mock_dependencies):
        """Tests the end-to-end flow from validation to evaluation."""
        df = create_synthetic_temporal_data(n_rows=60)
        
        # We need to mock the DataQualityAnalyzer to return a passing score
        with patch("core.feature_engineering.data_quality.DataQualityAnalyzer") as mock_qa:
            mock_qa.return_value.analyze.return_value = MagicMock(
                quality_score=0.9,
                status=True,
                issues=[],
                warnings=[]
            )
            
            pipeline = TemporalPipeline(
                dataframe=df,
                target_column="sales",
                datetime_column="timestamp",
                config=TemporalPipelineConfig(run_backtest=False) # Speed up test
            )
            
            # Execute
            result = pipeline.run()
            
            # Verify flow
            assert pipeline.is_fitted is True
            assert pipeline._lifecycle_state == "completed"
            assert "model_training" in result.metadata.step_timings
            assert "validation" in result.metadata.step_timings

    def test_circuit_breaker_trip(self):
        """
        Test that the REAL CircuitBreaker logic trips when 
        the REAL DataQualityAnalyzer is mocked to return failure.
        """
        # 1. Setup real data and real config
        df = create_synthetic_temporal_data(n_rows=50)
        config = TemporalPipelineConfig(min_quality_score=0.5) # Threshold is 0.5
        
        # 2. Only Mock the DataQualityAnalyzer return value
        with patch("core.feature_engineering.data_quality.DataQualityAnalyzer.analyze") as mock_analyze:
            # Simulate a very bad quality score
            mock_analyze.return_value = MagicMock(
                quality_score=0.1,  # 0.1 < 0.5 -> Should trigger!
                status=False,
                issues=["Critical failure"],
                warnings=[]
            )
            
            # 3. Use the REAL pipeline class
            pipeline = TemporalPipeline(
                dataframe=df,
                target_column="sales",
                datetime_column="timestamp",
                config=config
            )
            
            # 4. Assert that the exception IS raised
            with pytest.raises(CircuitBreakerTriggered):
                pipeline.run()
                
            # 5. Verify the state is 'failed'
            assert pipeline._lifecycle_state == "failed"

    def test_future_forecasting(self, mock_dependencies):
        """Tests the forecast_future method after fitting."""
        df = create_synthetic_temporal_data()
        
        with patch("core.feature_engineering.data_quality.DataQualityAnalyzer"):
            pipeline = TemporalPipeline(df, "sales", "timestamp")
            
            # Manually set state as if fitted to test prediction logic
            pipeline.is_fitted = True
            pipeline.model = mock_dependencies["prophet"].return_value
            
            forecast = pipeline.get_forecast_dataframe(periods=7)
            
            assert isinstance(forecast, pd.DataFrame)
            assert "yhat" in forecast.columns
            assert len(forecast) > 0

    def test_temporal_feature_engineering_integrity(self, mock_dependencies):
        """Ensures datetime column is correctly cast and sorted."""
        # Create unsorted data with string dates
        df = pl.DataFrame({
            "timestamp": ["2023-01-10", "2023-01-01", "2023-01-05"],
            "sales": [10, 20, 30]
        })
        
        pipeline = TemporalPipeline(df, "sales", "timestamp")
        pipeline._feature_engineering()
        
        # Check if it's now a datetime type
        assert pipeline.df["timestamp"].dtype in (pl.Date, pl.Datetime)
        
        # Check if it's sorted
        dates = pipeline.df["timestamp"].to_list()
        assert dates == sorted(dates)