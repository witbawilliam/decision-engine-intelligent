import pytest
import uuid
import polars as pl
from unittest.mock import MagicMock, patch
# FIX 1: Change 'task_validation' to 'tasks_validation'
from workers.tasks_validation import TaskValidator, TaskValidationConfig, ValidationError

class TestTaskValidator:

    @pytest.fixture
    def validator(self):
        config = TaskValidationConfig(min_quality_score=0.5)
        return TaskValidator(config=config)

    @pytest.fixture
    def mock_dfs(self):
        df = pl.DataFrame({"feature_a": [1.0, 2.0, 3.0], "feature_b": [10, 20, 30]})
        return df, df.clone()

    @pytest.fixture
    def valid_payload(self):
        return {
            "idempotency_key": str(uuid.uuid4()),
            "user_id": str(uuid.uuid4()),
            "filename": "test_data.csv"
        }

    # FIX 2: Update all patch decorators to use 'tasks_validation'
    @patch("workers.tasks_validation.V1.JobCreate")
    def test_gate_schema_failure(self, mock_v1, validator, mock_dfs):
        mock_v1.side_effect = Exception("Invalid Fields")
        ref_df, cur_df = mock_dfs

        result = validator.run(
            job_payload={"bad": "data"},
            reference_df=ref_df,
            current_df=cur_df,
            model=MagicMock(),
            feature_names=["feature_a"],
            quality_score=0.9,
            model_metric=0.8,
            elapsed_s=10.0
        )
        assert result.passed is False
        assert "ValidationError" in result.failure_reason

    @patch("workers.tasks_validation.PostgresClient.upsert")
    @patch("workers.tasks_validation.DriftDetector.check_drift")
    @patch("workers.tasks_validation.V1.JobCreate")
    def test_run_success_full_pipeline(self, mock_v1, mock_drift, mock_upsert, validator, mock_dfs, valid_payload):
        ref_df, cur_df = mock_dfs
        mock_v1.return_value = MagicMock()

        # SETUP: Mock high p-values (means LOW drift)
        mock_report = MagicMock()
        mock_report.is_drifted = False
        mock_report.flagged_features = []
        # 1.0 - 0.9 = 0.1 (Well below the 0.3 max_drift_score threshold)
        mock_report.drift_scores = {"feature_a": 0.9, "feature_b": 0.9}
        mock_drift.return_value = mock_report

        result = validator.run(
            job_payload=valid_payload,
            reference_df=ref_df,
            current_df=cur_df,
            model=MagicMock(),
            feature_names=["feature_a", "feature_b"],
            quality_score=0.9,      # Above 0.4
            model_metric=0.8,       # Above 0.5
            elapsed_s=10.0,         # Below 600s
            current_status="uploaded",
            target_status="validated"
        )

        assert result.passed is True, f"Failed: {result.failure_reason}"
        assert result.circuit_ok is True

    @patch("workers.tasks_validation.PostgresClient.upsert")
    def test_persistence_failure_non_fatal(self, mock_upsert, validator, mock_dfs, valid_payload):
        mock_upsert.side_effect = Exception("DB Down")
        ref_df, cur_df = mock_dfs

        result = validator.run(
            job_payload=valid_payload,
            reference_df=ref_df,
            current_df=cur_df,
            model=MagicMock(),
            feature_names=["feature_a"],
            quality_score=0.9,
            model_metric=0.8,
            elapsed_s=5.0
        )
        assert result.passed is True  # Business logic still passes
        assert any("Audit persistence failed" in w for w in result.warnings)