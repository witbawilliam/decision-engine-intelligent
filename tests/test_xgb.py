import pytest
import polars as pl
import numpy as np
import os
from pathlib import Path
from core.models.xgboost_model import XGBoostModel, ModelResult

# --- Fixtures ---

@pytest.fixture
def classification_data():
    """Generates a dataset with numeric and categorical features."""
    return pl.DataFrame({
        "feature_numeric": [1.0, 2.5, 3.0, 4.5, 5.0, 6.5, 7.0, 8.5],
        "feature_cat": ["A", "B", "A", "B", "A", "B", "A", "B"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1]
    })

@pytest.fixture
def regression_data():
    """Generates a continuous dataset for regression testing."""
    return pl.DataFrame({
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "target": [10.5, 20.0, 31.2, 39.8, 50.1]
    })

# --- Test Cases ---

def test_fit_and_predict_classification(classification_data):
    """Verify the end-to-end flow for binary classification."""
    model = XGBoostModel(problem_type="classification")
    model.fit(classification_data, target_column="target")
    
    # Predict on a subset (excluding target)
    test_df = classification_data.drop("target")
    result = model.predict(test_df)
    
    assert isinstance(result, ModelResult)
    assert result.predictions.shape[0] == 8
    assert result.probabilities is not None
    assert result.inference_time_ms > 0
    assert "feature_numeric" in result.feature_importance

def test_categorical_feature_handling(classification_data):
    """Verify that Utf8 columns are correctly handled via internal casting."""
    model = XGBoostModel(problem_type="classification")
    # This should not raise a ValueError despite 'feature_cat' being a string
    model.fit(classification_data, target_column="target")
    assert model._is_fitted is True

def test_model_persistence(regression_data, tmp_path):
    """Verify that saving and loading maintains model state and feature names."""
    model_path = tmp_path / "test_model.pkl"
    
    # Train and Save
    original_model = XGBoostModel(problem_type="regression")
    original_model.fit(regression_data, target_column="target")
    original_model.save(model_path)
    
    # Load
    loaded_model = XGBoostModel.load(model_path)
    
    assert loaded_model.problem_type == "regression"
    assert loaded_model.feature_names == ["x1"]
    assert loaded_model._is_fitted is True
    
    # Verify Loaded Predict
    test_df = regression_data.drop("target")
    original_preds = original_model.predict(test_df).predictions
    loaded_preds = loaded_model.predict(test_df).predictions
    
    np.testing.assert_array_almost_equal(original_preds, loaded_preds)

def test_unfitted_predict_error():
    """Verify the state guard prevents early inference."""
    model = XGBoostModel(problem_type="classification")
    
    # Matching the actual message in your implementation
    with pytest.raises(RuntimeError, match=r"Attempted to predict with an unfitted model"):
        model.predict(pl.DataFrame({"any_col": [1]}))

from core.models.xgboost_model import XGBoostModel, NotFittedError

def test_unfitted_predict_error():
    """Verify that we catch the specific NotFittedError."""
    model = XGBoostModel(problem_type="classification")
    
    # We test for the Exception Class itself
    with pytest.raises(NotFittedError):
        model.predict(pl.DataFrame({"any_col": [1]}))