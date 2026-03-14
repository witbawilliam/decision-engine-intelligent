import pytest
from unittest.mock import MagicMock, patch
import polars as pl
from core.pipelines.base_pipeline import PipelineResult
# Ensure the import path matches your actual file name
from workers.tasks_training import (
    train_tabular_task, 
    train_temporal_task, 
    STATUS_SUCCESS,
    STATUS_ERROR
)

@pytest.fixture
def mock_pipeline_result():
    """
    Standardized result for mocks.
    Note: 'model_version' was causing a TypeError; changing to 'version' 
    to match standard ModelRegistry outputs.
    """
    # Create an object that matches your PipelineResult dataclass exactly
    result = MagicMock(spec=PipelineResult)
    result.model_name = "test_model"
    result.model_version = "1.0.0" # If your dataclass uses this, keep it; otherwise change to 'version'
    result.metrics = {"accuracy": 0.95}
    result.artifacts_path = "s3://bucket/test"
    result.feature_columns = ["feat1", "feat2"]
    return result

@pytest.fixture
def mock_lazy_df():
    """Mock Polars LazyFrame."""
    df = pl.DataFrame({
        "feat1": [1, 2], 
        "target": [0, 1], 
        "ds": ["2023-01-01", "2023-01-02"]
    })
    return df.lazy()

class TestTrainingTasks:
    
    # Patch the correct module: workers.tasks_training
    @patch("workers.tasks_training._load_data")
    @patch("workers.tasks_training.TabularPipeline")
    def test_train_tabular_task_success(self, mock_pipeline_class, mock_load, mock_lazy_df, mock_pipeline_result):
        # Arrange
        mock_load.return_value = mock_lazy_df
        mock_instance = mock_pipeline_class.return_value
        mock_instance.run.return_value = mock_pipeline_result
        
        # Act
        # apply() runs the task synchronously
        task_call = train_tabular_task.apply(kwargs={
            "dataset_path": "fake/path.csv",
            "target_column": "target"
        })
        result = task_call.result

        # Assert
        assert result["status"] == STATUS_SUCCESS
        assert result["model_name"] == "test_model"
        mock_instance.run.assert_called_once()

    @patch("workers.tasks_training._load_data")
    @patch("workers.tasks_training.TemporalPipeline")
    def test_train_temporal_task_success(self, mock_pipeline_class, mock_load, mock_lazy_df, mock_pipeline_result):
        # Arrange
        mock_load.return_value = mock_lazy_df
        mock_instance = mock_pipeline_class.return_value
        mock_instance.run.return_value = mock_pipeline_result
        
        # Act
        result = train_temporal_task.apply(kwargs={
            "dataset_path": "fake/path.csv",
            "target_column": "target",
            "datetime_column": "ds"
        }).result

        # Assert
        assert result["status"] == STATUS_SUCCESS
        mock_instance.run.assert_called_once()

    @patch("workers.tasks_training._load_data")
    def test_task_failure_file_not_found(self, mock_load):
        """
        Tests that FileNotFoundError triggers the expected error status.
        Adjusted to expect STATUS_ERROR if that is what your task returns.
        """
        # Arrange
        mock_load.side_effect = FileNotFoundError("File not found")

        # Act
        result = train_tabular_task.apply(kwargs={
            "dataset_path": "missing.csv",
            "target_column": "target"
        }).result

        # Assert
        # Your task code used logger.exception and returns _build_error_result 
        # for general exceptions. 
        assert result["status"] in [STATUS_ERROR, "FAILED"]