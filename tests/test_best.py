import pytest
import polars as pl

from core.pipelines.base_pipeline import BasePipeline, ProblemType



# Dummy Concrete Pipeline for Testing


class DummyPipeline(BasePipeline):

    def _validate(self):
        if self.target_column not in self.df.columns:
            raise ValueError("Target missing")

    def _detect_problem_type(self):
        self.problem_type = ProblemType.REGRESSION

    def _feature_engineering(self):
        self.features = [c for c in self.df.columns if c != self.target_column]

    def _split(self):
        self.X_train = self.df.drop(self.target_column)
        self.y_train = self.df[self.target_column]
        self.X_test = self.X_train
        self.y_test = self.y_train

    def _train(self):
        self.model = object()  # Dummy model

    def _evaluate_final(self):
        return {"mae": 0.0}


# -------------------------------------------------
# Fixtures
# -------------------------------------------------

@pytest.fixture
def sample_df():
    return pl.DataFrame({
        "feature1": [1, 2, 3, 4],
        "target": [10, 20, 30, 40]
    })


# -------------------------------------------------
# Tests
# -------------------------------------------------

def test_pipeline_full_execution(sample_df):
    pipeline = DummyPipeline(
        dataframe=sample_df,
        target_column="target",
        experiment_id="test_exp"
    )

    result = pipeline.run()

    assert result.problem_type == ProblemType.REGRESSION
    assert "mae" in result.metrics
    assert result.metadata.duration > 0
    assert "validation" in result.metadata.step_timings
    assert pipeline.is_fitted is True


def test_double_run_protection(sample_df):
    pipeline = DummyPipeline(sample_df, "target")

    pipeline.run()

    with pytest.raises(RuntimeError):
        pipeline.run()


def test_missing_target_validation(sample_df):
    pipeline = DummyPipeline(sample_df, "not_existing")

    with pytest.raises(ValueError):
        pipeline.run()


def test_metadata_integrity(sample_df):
    pipeline = DummyPipeline(sample_df, "target")
    result = pipeline.run()

    metadata = result.metadata

    assert metadata.execution_id.startswith("default_exp")
    assert metadata.duration >= 0
    assert isinstance(metadata.step_timings, dict)
    assert "model_training" in metadata.step_timings


def test_artifact_structure(sample_df):
    pipeline = DummyPipeline(sample_df, "target")
    result = pipeline.run()

    assert isinstance(result.artifacts, dict)
    assert "model" in result.artifacts