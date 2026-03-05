import pytest
import polars as pl
import numpy as np
import pandas as pd

from core.pipelines.tabular_pipeline import TabularPipeline
from core.contracts.problem_type import ProblemType   # ✅ FIXED


# ==========================================================
# FIXTURE
# ==========================================================

@pytest.fixture
def sample_data():
    """
    Generates enterprise-style dataset with mixed types.
    """
    np.random.seed(42)
    rows = 120

    return pl.DataFrame({
        "age": np.random.randint(20, 60, rows),
        "salary": np.random.normal(50000, 15000, rows),
        "department": np.random.choice(["IT", "HR", "Sales", "Finance"], rows),
        "remote": np.random.choice([True, False], rows),
        "performance_score": np.random.uniform(0, 100, rows),
    })


# ==========================================================
# REGRESSION TEST
# ==========================================================

def test_regression_pipeline_execution(sample_data):

    pipeline = TabularPipeline(
        dataframe=sample_data,
        target_column="performance_score",
        experiment_id="exp_regression_test"
    )

    result = pipeline.run()

    # ----------------------------
    # Problem Type
    # ----------------------------
    assert pipeline.problem_type == ProblemType.REGRESSION

    # ----------------------------
    # Metrics
    # ----------------------------
    assert "mae" in result.metrics
    assert "rmse" in result.metrics
    assert isinstance(result.metrics["mae"], float)

    # ----------------------------
    # Metadata (BasePipeline telemetry)
    # ----------------------------
    assert "model_training" in result.metadata.step_timings
    assert result.metadata.execution_id.startswith("exp_regression_test")

    # ----------------------------
    # Drift Report Included
    # ----------------------------
    assert "drift_report" in result.artifacts
    assert isinstance(result.artifacts["drift_report"]["is_drifted"], bool)

    # ----------------------------
    # Feature Importance Included
    # ----------------------------
    assert "feature_importance" in result.artifacts


# ==========================================================
# CLASSIFICATION TEST
# ==========================================================

def test_classification_pipeline_execution(sample_data):

    df_class = sample_data.with_columns(
        (pl.col("performance_score") > 50)
        .cast(pl.Utf8)
        .alias("target")
    ).drop("performance_score")

    pipeline = TabularPipeline(
        dataframe=df_class,
        target_column="target",
        experiment_id="exp_classify_test"
    )

    result = pipeline.run()

    # ----------------------------
    # Problem Type
    # ----------------------------
    assert pipeline.problem_type == ProblemType.CLASSIFICATION

    # ----------------------------
    # Classification Metrics
    # ----------------------------
    assert "f1_weighted" in result.metrics
    assert "accuracy" in result.metrics
    assert isinstance(result.metrics["accuracy"], float)


# ==========================================================
# ARTIFACT INFERENCE TEST
# ==========================================================

def test_inference_with_raw_data_artifact(sample_data):

    pipeline = TabularPipeline(
        dataframe=sample_data,
        target_column="performance_score",
        experiment_id="exp_inference_test"
    )

    result = pipeline.run()

    model_artifact = result.artifacts["model"]

    # Raw row — no manual preprocessing
    raw_input = (
        sample_data
        .drop("performance_score")
        .head(1)
        .to_pandas()
    )

    try:
        prediction = model_artifact.predict(raw_input)

        assert len(prediction) == 1
        assert isinstance(prediction[0], (float, np.floating))

    except Exception as e:
        pytest.fail(
            f"Artifact failed to process raw data. "
            f"Preprocessing likely not embedded: {e}"
        )