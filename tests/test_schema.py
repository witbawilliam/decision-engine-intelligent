import pytest
import polars as pl
from datetime import datetime
from core.contracts.schema_inference import SchemaInference, ProblemType

# --- Fixtures ---

@pytest.fixture
def complex_df():
    """
    Generates a dataset with:
    - An ID column
    - A 'Fake' numeric (City Code) that is actually categorical
    - A temporal column
    - A binary target
    """
    return pl.DataFrame({
        "order_id": [1001, 1002, 1003, 1004, 1005],           # Identity
        "product_rating": [5, 4, 5, 2, 5],                    # Numeric
        "store_location_id": [1, 1, 2, 1, 2],                 # Semantic Category (Low Cardinality)
        "transaction_date": [datetime(2024, 1, i) for i in range(1, 6)], # Temporal
        "is_fraud": [0, 1, 0, 0, 0]                           # Target
    })

# --- Test Cases ---

def test_id_discovery(complex_df):
    """Verify that columns ending in _id with unique values are flagged as IDs."""
    inference = SchemaInference(complex_df)
    result = inference.infer()
    
    assert "order_id" in result.id_columns
    # store_location_id is NOT an ID because values repeat (Cardinality < N)
    assert "store_location_id" not in result.id_columns

def test_semantic_refinement(complex_df):
    """
    Verify the Cardinality-Ratio Heuristic: 
    store_location_id is numeric (Int), but should be inferred as Categorical.
    """
    inference = SchemaInference(complex_df)
    result = inference.infer()
    
    assert "store_location_id" in result.categorical_columns
    assert "store_location_id" not in result.numeric_columns

def test_problem_type_prioritization(complex_df):
    """
    Verify the Intent Priority Matrix:
    Even if targets look like classification, the presence of a date 
    should trigger 'Forecasting'.
    """
    inference = SchemaInference(complex_df)
    result = inference.infer()
    
    assert result.problem_type == ProblemType.FORECASTING

def test_regression_inference():
    """Verify that high-cardinality numeric targets trigger Regression."""
    # Create 25 unique values to cross the '20' threshold
    df = pl.DataFrame({
        "feature": list(range(25)),
        "target": [float(i) * 1.5 for i in range(25)]
    })
    inference = SchemaInference(df)
    result = inference.infer()
    
    assert result.problem_type == ProblemType.REGRESSION

def test_empty_dataframe_safety():
    """Ensure the engine raises a descriptive error on empty inputs."""
    df = pl.DataFrame()
    inference = SchemaInference(df)
    
    with pytest.raises(ValueError, match="Inference cannot run on an empty DataFrame"):
        inference.infer()

def test_classification_inference():
    """Verify that low-cardinality targets trigger Classification when no dates exist."""
    df = pl.DataFrame({
        "age": [20, 30, 40, 50, 60],
        "bought_item": ["Yes", "No", "Yes", "Yes", "No"]
    })
    inference = SchemaInference(df)
    result = inference.infer()
    
    assert result.problem_type == ProblemType.CLASSIFICATION