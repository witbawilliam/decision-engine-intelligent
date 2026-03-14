import pytest
from datetime import datetime, timezone
from app.schemas.job_schema import (
    V1, UUIDStr, ModelVersionStr, IdempotencyKeyStr, 
    ALLOWED_TRANSITIONS, JobStatus
)
from pydantic import TypeAdapter, ValidationError

# --- Test Annotated Types (Regex) ---

def test_uuid_validation():
    adapter = TypeAdapter(UUIDStr)
    # Valid
    assert adapter.validate_python("550e8400-e29b-41d4-a716-446655440000")
    # Invalid
    with pytest.raises(ValidationError):
        adapter.validate_python("not-a-uuid")

def test_model_version_validation():
    adapter = TypeAdapter(ModelVersionStr)
    assert adapter.validate_python("1.0.0")
    assert adapter.validate_python("2.3.11-rc1")
    with pytest.raises(ValidationError):
        adapter.validate_python("version-1")

def test_idempotency_key_validation():
    adapter = TypeAdapter(IdempotencyKeyStr)
    # Valid (printable ASCII, >16 chars)
    assert adapter.validate_python("a-very-long-unique-key-12345")
    # Invalid (contains space)
    with pytest.raises(ValidationError):
        adapter.validate_python("key with spaces")

# --- Test V1 Helper Methods ---

def test_validate_filename():
    # Valid
    assert V1.validate_filename("data_science_v1.csv") == "data_science_v1.csv"
    # Invalid characters
    with pytest.raises(ValueError, match="disallowed characters"):
        V1.validate_filename("data;drop table.csv")
    # Empty
    with pytest.raises(ValueError, match="must not be empty"):
        V1.validate_filename("   ")

def test_ensure_utc():
    # Aware datetime
    dt = datetime.now(timezone.utc)
    assert V1.ensure_utc(dt).tzinfo == timezone.utc
    # Naive datetime (should fail)
    naive_dt = datetime.now()
    with pytest.raises(ValueError, match="must be timezone-aware"):
        V1.ensure_utc(naive_dt)

# --- Test State Machine Logic ---

@pytest.mark.parametrize("current, next_state, expected", [
    ("uploaded", "validated", True),
    ("uploaded", "failed", True),
    ("running", "completed", True),
    ("completed", "running", False),  # Terminal state cannot move
    ("running", "uploaded", False),   # Cannot go backwards
])
def test_state_transitions(current, next_state, expected):
    allowed = ALLOWED_TRANSITIONS.get(current, frozenset())
    assert (next_state in allowed) is expected