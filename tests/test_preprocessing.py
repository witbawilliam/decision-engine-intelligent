import pytest
import polars as pl
import numpy as np
from core.feature_engineering.preprocessing_utils import FeatureProcessor, ImputationStrategy

# --- Fixtures ---

@pytest.fixture
def messy_df():
    """
    A dataset with various enterprise 'problems':
    1. Missing values (nulls)
    2. High correlation (leakage)
    3. Low variance (constant column)
    4. Duplicates
    """
    return pl.DataFrame({
        "id": [1, 2, 2, 3, 4, 5],                  # Contains a duplicate row
        "target": [10, 20, 20, 30, 40, 50],        # Regression target
        "leaked_feature": [10.1, 20.1, 20.1, 30.1, 40.1, 50.1], # 0.99+ correlation
        "constant_col": [1, 1, 1, 1, 1, 1],        # Zero variance
        "null_feature": [1.0, None, None, 4.0, 5.0, 6.0], # Needs median imputation
        "cat_feature": ["A", "B", "B", None, "A", "B"]    # Needs mode imputation
    })

# --- Test Cases ---

def test_full_pipeline_execution(messy_df):
    """Verify that the processor runs the full pipeline without crashing."""
    processor = FeatureProcessor(target_column="target")
    clean_df, metadata = processor.process(messy_df)
    
    # Check shape (Original 6 rows -> 5 unique; 6 cols -> dropped leaked and constant)
    assert clean_df.height == 5
    assert "constant_col" not in clean_df.columns
    assert "leaked_feature" not in clean_df.columns
    
    # Check Metadata
    assert metadata.duplicates_removed == 1
    assert "leaked_feature" in metadata.leaked_features
    assert "constant_col" in metadata.dropped_features

def test_imputation_logic(messy_df):
    """Verify that nulls are filled with correct statistical strategies."""
    processor = FeatureProcessor(target_column="target", numeric_strategy=ImputationStrategy.MEDIAN)
    clean_df, _ = processor.process(messy_df)
    
    # null_feature values: [1.0, 4.0, 5.0, 6.0] -> Median is 4.5
    # The nulls in indices 1 and 2 (duplicate removed) should be 4.5
    assert clean_df["null_feature"].null_count() == 0
    assert clean_df["null_feature"][1] == 4.5

def test_categorical_encoding(messy_df):
    """Verify strings are converted to physical integers."""
    processor = FeatureProcessor(target_column="target")
    clean_df, _ = processor.process(messy_df)
    
    # 'cat_feature' should now be a numeric type (physical representation of categorical)
    assert clean_df["cat_feature"].dtype.is_integer()

def test_empty_dataframe_error():
    """Ensure the system fails fast on empty inputs."""
    empty_df = pl.DataFrame({"a": []})
    processor = FeatureProcessor(target_column="target")
    with pytest.raises(ValueError, match="zero records"):
        processor.process(empty_df)