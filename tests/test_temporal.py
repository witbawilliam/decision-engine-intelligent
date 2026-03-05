import pytest
import polars as pl
import numpy as np
from datetime import datetime
from core.feature_engineering.temporal_features import TemporalFeatureEngineer, TemporalFeatureConfig, TemporalResolution

# --- Fixtures ---

@pytest.fixture
def temporal_df():
    """Generates a dataset covering key temporal boundaries."""
    return pl.DataFrame({
        "transaction_date": [
            datetime(2024, 1, 1, 12, 0),   # Monday, Mid-day, Start of year
            datetime(2024, 6, 15, 18, 30),  # Saturday (Weekend), Mid-year
            datetime(2024, 12, 31, 23, 59)  # Tuesday, Year-end
        ]
    })

# --- Test Cases ---

def test_basic_decomposition(temporal_df):
    """Verify standard date part extraction (Year, Month, Day)."""
    engineer = TemporalFeatureEngineer()
    result = engineer.transform(temporal_df, ["transaction_date"])
    
    # Check if expected columns exist
    expected_cols = ["transaction_date_year", "transaction_date_month", "transaction_date_day"]
    for col in expected_cols:
        assert col in result.columns
    
    # Verify specific values (Row 0: Jan 1st)
    assert result["transaction_date_month"][0] == 1
    assert result["transaction_date_day"][0] == 1

def test_weekend_detection(temporal_df):
    """Verify the boolean logic: Saturday/Sunday == True."""
    engineer = TemporalFeatureEngineer(config=TemporalFeatureConfig(add_is_weekend=True))
    result = engineer.transform(temporal_df, ["transaction_date"])
    
    # Jan 1 2024 is Monday (False), June 15 2024 is Saturday (True)
    assert result["transaction_date_is_weekend"][0] is False
    assert result["transaction_date_is_weekend"][1] is True

def test_cyclic_math_precision(temporal_df):
    """
    Verify the Sine/Cosine formula: x_sin = sin(2*pi*month/12).
    Formula Name: Trigonometric Continuity Validation
    """
    engineer = TemporalFeatureEngineer(config=TemporalFeatureConfig(add_cyclic_signals=True))
    result = engineer.transform(temporal_df, ["transaction_date"])
    
    # For Month 6 (June): sin(2 * pi * 6 / 12) = sin(pi) = 0
    # Note: Float precision might yield a near-zero value
    june_sin = result["transaction_date_month_sin"][1]
    assert np.isclose(june_sin, 0.0, atol=1e-7)
    
    # For Month 12 (Dec): cos(2 * pi * 12 / 12) = cos(2*pi) = 1
    dec_cos = result["transaction_date_month_cos"][2]
    assert np.isclose(dec_cos, 1.0, atol=1e-7)

def test_high_resolution_extraction(temporal_df):
    """Verify hour/minute extraction when resolution is set to HIGH."""
    config = TemporalFeatureConfig(resolution=TemporalResolution.HIGH)
    engineer = TemporalFeatureEngineer(config=config)
    result = engineer.transform(temporal_df, ["transaction_date"])
    
    assert "transaction_date_hour" in result.columns
    assert result["transaction_date_hour"][2] == 23
    assert result["transaction_date_minute"][2] == 59

def test_empty_column_list(temporal_df):
    """Ensure the engineer handles empty inputs gracefully (Identity Mapping)."""
    engineer = TemporalFeatureEngineer()
    result = engineer.transform(temporal_df, [])
    
    # Result should be identical to input if no columns are transformed
    assert result.equals(temporal_df)