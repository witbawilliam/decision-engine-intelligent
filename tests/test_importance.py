import pytest
import numpy as np
import polars as pl
from xgboost import XGBRegressor
from core.evaluation.feature_importance import XGBExplainer, ExplanationResult

@pytest.fixture
def trained_model_and_data():
    """
    Creates a model where:
    target = 10 * feature_A + 1 * feature_B + 0 * noise
    """
    np.random.seed(42)
    X = np.random.rand(100, 3)
    # Feature A is 10x more important than B; C is noise
    y = 10 * X[:, 0] + 1 * X[:, 1] + np.random.normal(0, 0.01, 100)
    
    feature_names = ["feature_A", "feature_B", "feature_noise"]
    df = pl.DataFrame(X, schema=feature_names)
    
    model = XGBRegressor(n_estimators=10, max_depth=3, learning_rate=0.1)
    model.fit(X, y)
    
    return model, df, feature_names

def test_explain_logic_consistency(trained_model_and_data):
    model, df, feature_names = trained_model_and_data
    explainer = XGBExplainer(model, feature_names)
    
    # Run explanation on a subset
    test_df = df.head(10)
    result = explainer.explain(test_df)
    
    # 1. Check Output Type
    assert isinstance(result, ExplanationResult)
    
    # 2. Verify Importance Ranking (feature_A should be top)
    top_shap_feature = next(iter(result.shap_importance))
    top_gain_feature = next(iter(result.gain_importance))
    
    assert top_shap_feature == "feature_A"
    assert top_gain_feature == "feature_A"
    
    # 3. Verify Noise Feature has lowest importance
    assert result.shap_importance["feature_noise"] < result.shap_importance["feature_B"]

def test_shap_additivity(trained_model_and_data):
    """
    Validation: Expected Value + Sum(SHAP) should equal Model Prediction.
    This is the core mathematical requirement for 'True' explanations.
    """
    model, df, feature_names = trained_model_and_data
    explainer = XGBExplainer(model, feature_names)
    
    test_instance = df.head(1)
    result = explainer.explain(test_instance)
    
    # Sum of SHAP values + Base Value (Expected Value)
    sum_shap = result.shap_values[0].sum()
    total_prediction = result.expected_value + sum_shap
    
    # Actual model prediction
    actual_pred = model.predict(test_instance.to_numpy())[0]
    
    # Assert they are equal (within floating point tolerance)
    assert np.isclose(total_prediction, actual_pred, atol=1e-5)

def test_polars_conversion(trained_model_and_data):
    model, df, feature_names = trained_model_and_data
    explainer = XGBExplainer(model, feature_names)
    
    result = explainer.explain(df.head(5))
    shap_df = result.to_polars()
    
    assert isinstance(shap_df, pl.DataFrame)
    assert shap_df.columns == feature_names
    assert shap_df.height == 5