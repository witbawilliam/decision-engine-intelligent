import pytest
import uuid
from workers.tasks_validation import TaskValidator

def test_validator_instantiation():
    """Verifies the class can be created with default config."""
    validator = TaskValidator()
    assert validator.config.run_explainability is True

def test_validation_path_minimal_coverage_fixed():
    """Triggers the Gate 1 error by providing a dict missing required UUIDs."""
    validator = TaskValidator()
    # Providing an empty dict triggers the ValidationError and the ErrorLogger
    result = validator.run(
        job_payload={}, 
        reference_df=None, 
        current_df=None, 
        model=None, 
        feature_names=[], 
        quality_score=0, 
        model_metric=0, 
        elapsed_s=0
    )
    assert result.passed is False
    assert "ValidationError" in result.failure_reason