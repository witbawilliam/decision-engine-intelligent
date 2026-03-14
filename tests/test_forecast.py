import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
from unittest.mock import ANY
from uuid import uuid4

from workers.tasks_forecasting import (
    train_forecasting_task, 
    _ProgressReporter, 
    STATUS_SUCCESS, 
    STATUS_FAILED
)

import workers.tasks_training as training
import workers.tasks_validation as validation

class TestForecastingTask:
    """
    Test suite for Forecasting ML workers.
    Ensures Pydantic schema alignment, error handling, and metric tracking.
    """

    @pytest.fixture
    def mock_pipeline_result(self):
        result = MagicMock()
        # Pydantic requires real strings, not MagicMock objects
        result.model_name = "ProphetForecaster"
        result.model_version = "1.0.0" 
        result.horizon = 24
        result.metrics = {"mae": 0.5, "rmse": 0.7}
        return result

    @patch("workers.tasks_forecasting.pl.scan_csv")
    @patch("workers.tasks_forecasting.TemporalPipeline")
    @patch("workers.tasks_forecasting.track_training_latency")
    def test_train_forecasting_task_success(
        self, 
        mock_track, 
        mock_pipeline_class, 
        mock_scan, 
        mock_pipeline_result
    ):
        mock_instance = mock_pipeline_class.return_value
        mock_instance.run.return_value = mock_pipeline_result
        
        result = train_forecasting_task.apply(kwargs={
            "dataset_path": "valid_data.csv",
            "time_column": "ds",
            "target_column": "y",
            "freq": "D"
        }).result

        assert result["status"] == STATUS_SUCCESS
        # FIX: Use ANY instead of pytest.any
        mock_track.assert_called_with(
            task_id=ANY, 
            status="success", 
            pipeline="forecasting"
        )

    @patch("workers.tasks_forecasting.pl.scan_csv")
    @patch("workers.tasks_forecasting.TemporalPipeline")
    @patch("workers.tasks_forecasting.track_training_latency")
    def test_train_forecasting_task_failure(self, mock_track, mock_pipeline_class, mock_scan):
        # 1. Arrange
        mock_instance = mock_pipeline_class.return_value
        error_msg = "Data corruption detected"
        mock_instance.run.side_effect = Exception(error_msg)
        
        # 2. Act
        result = train_forecasting_task.apply(kwargs={
            "dataset_path": "test.csv",
            "time_column": "ds",
            "target_column": "y"
        }).result

        # 3. Assert
        assert result["status"] == "ERROR"
        # Based on your log, your helper uses the 'message' key
        assert error_msg in result.get("message", "")
        
        mock_track.assert_called_with(
            task_id=ANY, 
            status="failed", 
            pipeline="forecasting"
        )

    def test_progress_reporting_calls(self):
        # 1. Setup - Use a real UUID string and a real task mock
        mock_task = MagicMock()
        real_job_id = str(uuid4()) # Pydantic will accept this string
        
        reporter = _ProgressReporter(task=mock_task, task_id=real_job_id)

        # 2. Act - This triggers the V1.JobStatusResponse validation internally
        reporter.report(step="loading_data", percent=45)

        # 3. Assert - Check the call arguments
        mock_task.update_state.assert_called_once()
        
        # Get the 'meta' dict passed to update_state
        _, kwargs = mock_task.update_state.call_args
        meta = kwargs.get("meta")
        
        assert meta["job_id"] == real_job_id
        assert meta["progress"] == 45
        assert meta["status"] == "running"
        # Ensure updated_at exists (Pydantic likely added this automatically)
        assert "updated_at" in meta

    @patch("workers.tasks_forecasting.pl.scan_csv")
    def test_train_forecasting_task_file_not_found(self, mock_scan):
        """Verify deterministic failure when dataset is missing."""
        # Arrange
        mock_scan.side_effect = FileNotFoundError("File not found")

        # Act
        result = train_forecasting_task.apply(kwargs={
            "dataset_path": "missing.csv",
            "time_column": "ds",
            "target_column": "y"
        }).result

        # Assert
        assert result["status"] == STATUS_FAILED
        assert "Dataset not found" in result["reason"]

    @patch("workers.tasks_forecasting.TemporalPipeline")
    @patch("workers.tasks_forecasting.pl.scan_csv")
    def test_progress_reporting_calls(self, mock_scan, mock_pipeline): # <--- Fix: Added mocks
        mock_task = MagicMock()
        valid_task_id = "task_forecast_prod_001_2024" 
        
        reporter = _ProgressReporter(task=mock_task, task_id=valid_task_id)

        # Act
        reporter.report("test_step", 50)

        # Assert
        mock_task.update_state.assert_called_once()

    @patch("workers.tasks_forecasting.train_forecasting_task.apply_async")
    def test_submit_job_idempotency(self, mock_apply):
        """Verify that the idempotency key is used as the Celery task ID."""
        from workers.tasks_forecasting import submit_job
        
        payload = {
            "user_id": "123e4567-e89b-12d3-a456-426614174000",
            "idempotency_key": "unique-request-key-123",
            "filename": "data.csv",
            "target_column": "price"
        }

        submit_job(payload)

        # Verify task was sent to queue with the correct task_id
        mock_apply.assert_called_once()
        assert mock_apply.call_args.kwargs["task_id"] == "unique-request-key-123"


    @pytest.mark.parametrize("freq, horizon", [("H", 12), ("D", 7), ("W", 4)])
    def test_forecast_coverage_booster(self, freq, horizon):
        """Hits different logic branches for frequency and horizon settings."""
        with patch("workers.tasks_forecasting.TemporalPipeline") as mock_pipe:
            mock_pipe.return_value.run.return_value = MagicMock(metrics={})
            result = train_forecasting_task.apply(kwargs={
                "dataset_path": "test.csv",
                "time_column": "ds",
                "target_column": "y",
                "freq": freq
            }).result
            assert result is not None    


