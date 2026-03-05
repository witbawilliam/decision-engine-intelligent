import pytest
import polars as pl
from datetime import datetime
from core.labeling.target_identifier import TargetIdentifier, ProblemType

# --- Fixtures ---

@pytest.fixture
def marketing_df():
    """
    Standard dataset with typical 'noise':
    - An ID at the start.
    - A 'Label' column (Target).
    - A random float feature.
    """
    return pl.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "age": [25, 30, 35, 40, 45],
        "spend_score": [0.1, 0.5, 0.8, 0.2, 0.9],
        "outcome": [0, 1, 0, 0, 1]  # Target hint 'outcome'
    })

@pytest.fixture
def forecasting_df():
    """Dataset designed to trigger forecasting logic."""
    return pl.DataFrame({
        "date": [datetime(2024, 1, i) for i in range(1, 6)],
        "store_id": [10, 10, 10, 10, 10],
        "sales_volume": [150.5, 200.0, 180.5, 220.0, 210.0]
    })

# --- Test Cases ---

def test_hint_keyword_priority(marketing_df):
    """Verify that 'outcome' is picked over other columns due to name weights."""
    identifier = TargetIdentifier(marketing_df)
    result = identifier.identify()
    
    assert result.target_column == "outcome"
    assert result.problem_type == ProblemType.BINARY
    assert result.confidence >= 0.6  # High confidence due to name match

def test_forbidden_keyword_exclusion(marketing_df):
    """Ensure customer_id is never picked, even if it has high variance."""
    identifier = TargetIdentifier(marketing_df)
    result = identifier.identify()
    
    # customer_id should not even be in the candidate scores if weights are tuned
    assert "customer_id" not in result.metadata.get("scores", {})

def test_forecasting_detection(forecasting_df):
    """Verify that numeric targets + date columns = Forecasting."""
    identifier = TargetIdentifier(forecasting_df)
    result = identifier.identify()
    
    # Should pick sales_volume as target and detect forecasting
    assert result.target_column == "sales_volume"
    assert result.problem_type == ProblemType.FORECASTING

def test_manual_override():
    """Verify that forcing a target ignores heuristics."""
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    # Heuristically 'b' might be picked (position), but we force 'a'
    identifier = TargetIdentifier(df, force_target="a")
    result = identifier.identify()
    
    assert result.target_column == "a"
    assert result.metadata["source"] == "override"
    assert result.confidence == 1.0

def test_regression_for_floats():
    """Verify that float columns trigger regression even with few rows."""
    df = pl.DataFrame({
        "feature": [1, 2, 3],
        "price": [100.50, 200.75, 150.00]
    })
    identifier = TargetIdentifier(df)
    result = identifier.identify()
    
    assert result.target_column == "price"
    assert result.problem_type == ProblemType.REGRESSION

def test_binary_vs_multiclass():
    """Verify the distinction between Binary and Multi-class classification."""
    # Binary
    df_bin = pl.DataFrame({"x": [1, 2, 3], "target": [0, 1, 0]})
    assert TargetIdentifier(df_bin).identify().problem_type == ProblemType.BINARY
    
    # Multi-class (3 unique values)
    df_multi = pl.DataFrame({"x": [1, 2, 3], "target": ["A", "B", "C"]})
    assert TargetIdentifier(df_multi).identify().problem_type == ProblemType.CLASSIFICATION