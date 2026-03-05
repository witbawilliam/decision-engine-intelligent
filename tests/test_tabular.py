import polars as pl
import pytest

from core.feature_engineering.tabular_features import (
    TabularIntelligenceEngine,
)


# ==========================================================
# TEST FEATURE CLASSIFICATION
# ==========================================================

def test_feature_classification():

    df = pl.DataFrame({
        "age": [25, 30, 40],
        "income": [50000, 60000, 70000],
        "city": ["Buea", "Douala", "Yaounde"],
        "target": [1, 0, 1],
    })

    engine = TabularIntelligenceEngine(df, target_column="target")
    metadata = engine.classify_features()

    assert "age" in metadata.numeric
    assert "income" in metadata.numeric
    assert "city" in metadata.categorical
    assert metadata.target == "target"
    assert metadata.problem_type == "classification"


# ==========================================================
# TEST REGRESSION DETECTION
# ==========================================================

def test_regression_detection():

    df = pl.DataFrame({
        "feature": [1, 2, 3, 4],
        "target": [10.5, 20.1, 30.2, 40.3],
    })

    engine = TabularIntelligenceEngine(df, target_column="target")
    metadata = engine.classify_features()

    assert metadata.problem_type == "regression"


# ==========================================================
# TEST FORECASTING DETECTION
# ==========================================================

def test_forecasting_detection():

    df = pl.DataFrame({
        "date": pl.date_range(
            start=pl.date(2024, 1, 1),
            end=pl.date(2024, 1, 4),
            interval="1d",
            eager=True
        ),
        "target": [100, 120, 130, 140],
    })

    engine = TabularIntelligenceEngine(df, target_column="target")
    metadata = engine.classify_features()

    assert metadata.problem_type == "forecasting"


# ==========================================================
# TEST DATETIME FEATURE ENGINEERING
# ==========================================================

def test_datetime_feature_engineering():

    df = pl.DataFrame({
        "date": pl.date_range(
            start=pl.date(2024, 1, 1),
            end=pl.date(2024, 1, 3),
            interval="1d",
            eager=True
        ),
        "target": [1, 2, 3],
    })

    engine = TabularIntelligenceEngine(df, target_column="target")
    engineered = engine.engineer_features()

    assert "date_year" in engineered.columns
    assert "date_month" in engineered.columns
    assert "date_day" in engineered.columns
    assert "date_weekday" in engineered.columns


# ==========================================================
# TEST NUMERIC INTERACTION FEATURE
# ==========================================================

def test_numeric_interaction_feature():

    df = pl.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "target": [0, 1, 0],
    })

    engine = TabularIntelligenceEngine(df, target_column="target")
    engineered = engine.engineer_features()

    assert "a_x_b" in engineered.columns


# ==========================================================
# TEST SAFE LOG TRANSFORM
# ==========================================================

def test_log_transform():

    df = pl.DataFrame({
        "positive_feature": [1, 10, 100],
        "target": [0, 1, 0],
    })

    engine = TabularIntelligenceEngine(df, target_column="target")
    engineered = engine.engineer_features()

    assert "positive_feature_log" in engineered.columns


# ==========================================================
# TEST LOG NOT APPLIED TO NEGATIVE VALUES
# ==========================================================

def test_log_not_applied_to_negative():

    df = pl.DataFrame({
        "mixed_feature": [-1, 2, 3],
        "target": [0, 1, 0],
    })

    engine = TabularIntelligenceEngine(df, target_column="target")
    engineered = engine.engineer_features()

    assert "mixed_feature_log" not in engineered.columns


# ==========================================================
# TEST TARGET NOT MODIFIED
# ==========================================================

def test_target_column_preserved():

    df = pl.DataFrame({
        "feature": [1, 2, 3],
        "target": [10, 20, 30],
    })

    engine = TabularIntelligenceEngine(df, target_column="target")
    engineered = engine.engineer_features()

    assert "target" in engineered.columns