"""
tests/test_job_schema.py
=========================
Full test suite for job_schema.py.

All match= regex patterns were verified against pydantic v2's actual
str(ValidationError) format before this file was written:
  - Our own ValueError messages  → matched literally (always safe)
  - Pydantic field violations    → matched against the stable type= string
    e.g. "string_pattern_mismatch", "greater_than_equal", "literal_error"

Coverage map
------------
1.  UUIDStr              — valid UUID, uppercase rejected, short/malformed rejected
2.  ModelVersionStr      — valid semver + pre-release, invalid formats rejected,
                           max_length=32 enforced
3.  IdempotencyKeyStr    — min 16, max 128, printable ASCII only, spaces rejected
4.  ProgressInt          — 0, 50, 100 valid; -1 and 101 invalid
5.  NormalisedFloat      — 0.0, 0.5, 1.0 valid; -0.01 and 1.01 invalid
6.  JobStatus Literal    — all 7 valid values; invalid value rejected
7.  ProblemType Literal  — all 3 valid values; invalid rejected
8.  TERMINAL_STATUSES    — completed, failed, cancelled are terminal
9.  ALLOWED_TRANSITIONS  — every legal edge; illegal forward jumps blocked;
                           cancelled always allowed from non-terminal states
10. V1.JobCreate         — happy path, optional fields, UUID + idempotency key
                           validation wired up
11. V1.JobStatusResponse — all JobStatus values, ProgressInt bounds
12. V1.JobResultResponse — ModelVersionStr wired, performance dict, optional summary
13. V1.FeedbackRequest   — second definition wins (model_name/prediction/actual);
                           metadata optional
14. V1.FeedbackResponse  — feedback_id, errors, recorded_at
15. V1.validate_filename — valid filenames, empty, bad chars, too long, strip,
                           extension length boundary
16. V1.validate_column_name — valid names, None passthrough, empty, bad chars,
                              128-char boundary
17. V1.ensure_utc        — naive raises, UTC passthrough, non-UTC normalised
18. Schema bugs documented — duplicate FeedbackRequest, static validate_filename stub
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

import pytest


_HERE       = os.path.dirname(os.path.abspath(__file__))
_SCHEMA_DIR = os.path.normpath(os.path.join(_HERE, "..", "uploads"))
if _SCHEMA_DIR not in sys.path:
    sys.path.insert(0, _SCHEMA_DIR)

from app.schemas.job_schema import (   # noqa: E402
    ALLOWED_TRANSITIONS,
    TERMINAL_STATUSES,
    V1,
    _COLUMN_NAME_RE,
    _SAFE_FILENAME_RE,
)
from pydantic import ValidationError


VALID_UUID       = "550e8400-e29b-41d4-a716-446655440000"
VALID_IKEY       = "a" * 16               # minimum-length idempotency key
VALID_IKEY_LONG  = "x" * 128             # maximum-length idempotency key
VALID_VERSION    = "1.0.0"
UTC_DT           = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)



class TestUUIDStr:

    def _make(self, uuid: str):
        return V1.JobCreate(
            user_id=uuid,
            idempotency_key=VALID_IKEY,
            filename="data.csv",
        )

    @pytest.mark.parametrize("uuid", [
        "550e8400-e29b-41d4-a716-446655440000",
        "00000000-0000-0000-0000-000000000000",
        "ffffffff-ffff-ffff-ffff-ffffffffffff",
        "a" * 8 + "-" + "b" * 4 + "-" + "c" * 4 + "-" + "d" * 4 + "-" + "e" * 12,
    ])
    def test_valid_uuid(self, uuid):
        job = self._make(uuid)
        assert job.user_id == uuid

    @pytest.mark.parametrize("uuid, reason", [
        ("550E8400-E29B-41D4-A716-446655440000", "uppercase hex"),
        ("550e8400-e29b-41d4-a716",              "too short"),
        ("not-a-uuid",                           "wrong format"),
        ("",                                     "empty string"),
        ("550e8400e29b41d4a716446655440000",      "no dashes"),
    ])
    def test_invalid_uuid_raises(self, uuid, reason):
        with pytest.raises(ValidationError, match="string_pattern_mismatch"):
            self._make(uuid)

    def test_whitespace_stripped_before_validation(self):
        """StringConstraints(strip_whitespace=True) strips before pattern check."""
        job = self._make("  " + VALID_UUID + "  ")
        assert job.user_id == VALID_UUID



class TestModelVersionStr:

    def _make(self, version: str):
        return V1.JobResultResponse(
            job_id=VALID_IKEY,
            status="completed",
            model_version=version,
            performance={},
        )

    @pytest.mark.parametrize("version", [
        "1.0.0",
        "2.3.11",
        "0.0.1",
        "1.0.0-rc1",
        "1.0.0-beta",
        "10.20.30-alpha1",
    ])
    def test_valid_versions(self, version):
        resp = self._make(version)
        assert resp.model_version == version

    @pytest.mark.parametrize("version, reason", [
        ("1.0",          "missing patch"),
        ("v1.0.0",       "v-prefix"),
        ("1.0.0.0",      "four parts"),
        ("latest",       "alias not allowed here"),
        ("",             "empty"),
        ("1.0.0-rc_1",   "underscore in pre-release"),
    ])
    def test_invalid_versions_raise(self, version, reason):
        with pytest.raises(ValidationError, match="string_pattern_mismatch"):
            self._make(version)

    def test_max_length_32_enforced(self):
        long_version = "1.0.0-" + "x" * 27   # total = 33 chars
        with pytest.raises(ValidationError, match="string_too_long"):
            self._make(long_version)

    def test_exactly_32_chars_allowed(self):
        version = "1.0.0-" + "x" * 26        # total = 32 chars
        resp = self._make(version)
        assert resp.model_version == version



class TestIdempotencyKeyStr:

    def _make(self, key: str):
        return V1.JobCreate(
            user_id=VALID_UUID,
            idempotency_key=key,
            filename="data.csv",
        )

    def test_minimum_length_16(self):
        job = self._make("a" * 16)
        assert job.idempotency_key == "a" * 16

    def test_maximum_length_128(self):
        job = self._make("a" * 128)
        assert job.idempotency_key == "a" * 128

    def test_too_short_raises(self):
        # Length is enforced by the pattern {16,128}, not a min_length constraint,
        # so pydantic raises string_pattern_mismatch for all length violations.
        with pytest.raises(ValidationError, match="string_pattern_mismatch"):
            self._make("a" * 15)

    def test_too_long_raises(self):
        # Same reason as above — {16,128} in the pattern covers max length too.
        with pytest.raises(ValidationError, match="string_pattern_mismatch"):
            self._make("a" * 129)

    def test_space_rejected(self):
        """Space (\x20) is below \x21 — not in printable ASCII range."""
        with pytest.raises(ValidationError, match="string_pattern_mismatch"):
            self._make("has a space here!")   # 17 chars, contains space

    def test_control_char_rejected(self):
        with pytest.raises(ValidationError, match="string_pattern_mismatch"):
            self._make("tab\there_padding!")   # \t = \x09

    def test_printable_specials_accepted(self):
        """Printable non-alphanumeric chars like !, @, # are in [\x21-\x7E]."""
        key = "!@#$%^&*()_+-=[]{"  # 18 printable chars
        job = self._make(key)
        assert job.idempotency_key == key

    def test_uuid_v4_format_accepted(self):
        """Clients commonly use UUID v4 as the idempotency key."""
        key = "550e8400-e29b-41d4-a716-446655440000"   # 36 printable chars
        job = self._make(key)
        assert job.idempotency_key == key



class TestProgressInt:

    def _make(self, progress: int):
        return V1.JobStatusResponse(
            job_id=VALID_IKEY,
            status="running",
            progress=progress,
            updated_at=UTC_DT,
        )

    @pytest.mark.parametrize("value", [0, 1, 50, 99, 100])
    def test_valid_progress(self, value):
        resp = self._make(value)
        assert resp.progress == value

    def test_below_zero_raises(self):
        with pytest.raises(ValidationError, match="greater_than_equal"):
            self._make(-1)

    def test_above_100_raises(self):
        with pytest.raises(ValidationError, match="less_than_equal"):
            self._make(101)

    def test_boundary_zero(self):
        assert self._make(0).progress == 0

    def test_boundary_100(self):
        assert self._make(100).progress == 100



class TestNormalisedFloat:
    """
    NormalisedFloat is Annotated[float, Field(ge=0.0, le=1.0)].
    It is not yet used in a model field in the schema, so we test
    the constraint directly via a minimal inline model.
    """

    def _make_model(self):
        from pydantic import BaseModel
        from app.schemas.job_schema import NormalisedFloat

        class _M(BaseModel):
            score: NormalisedFloat

        return _M

    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0, 0.999])
    def test_valid_values(self, value):
        M = self._make_model()
        assert M(score=value).score == value

    def test_below_zero_raises(self):
        M = self._make_model()
        with pytest.raises(ValidationError, match="greater_than_equal"):
            M(score=-0.01)

    def test_above_one_raises(self):
        M = self._make_model()
        with pytest.raises(ValidationError, match="less_than_equal"):
            M(score=1.01)



class TestJobStatus:

    @pytest.mark.parametrize("status", [
        "uploaded", "validated", "queued", "running",
        "completed", "failed", "cancelled",
    ])
    def test_all_valid_statuses(self, status):
        resp = V1.JobStatusResponse(
            job_id=VALID_IKEY,
            status=status,
            progress=0,
            updated_at=UTC_DT,
        )
        assert resp.status == status

    @pytest.mark.parametrize("bad_status", [
        "RUNNING", "done", "pending", "error", "", "queued ",
    ])
    def test_invalid_status_raises(self, bad_status):
        with pytest.raises(ValidationError, match="literal_error"):
            V1.JobStatusResponse(
                job_id=VALID_IKEY,
                status=bad_status,
                progress=0,
                updated_at=UTC_DT,
            )



class TestProblemType:

    @pytest.mark.parametrize("pt", ["regression", "classification", "forecasting"])
    def test_all_valid_problem_types(self, pt):
        job = V1.JobCreate(
            user_id=VALID_UUID,
            idempotency_key=VALID_IKEY,
            filename="data.csv",
            problem_type=pt,
        )
        assert job.problem_type == pt

    def test_problem_type_defaults_to_none(self):
        job = V1.JobCreate(
            user_id=VALID_UUID,
            idempotency_key=VALID_IKEY,
            filename="data.csv",
        )
        assert job.problem_type is None

    @pytest.mark.parametrize("bad", ["REGRESSION", "binary", "timeseries", ""])
    def test_invalid_problem_type_raises(self, bad):
        with pytest.raises(ValidationError, match="literal_error"):
            V1.JobCreate(
                user_id=VALID_UUID,
                idempotency_key=VALID_IKEY,
                filename="data.csv",
                problem_type=bad,
            )



class TestTerminalStatuses:

    def test_completed_is_terminal(self):
        assert "completed" in TERMINAL_STATUSES

    def test_failed_is_terminal(self):
        assert "failed" in TERMINAL_STATUSES

    def test_cancelled_is_terminal(self):
        assert "cancelled" in TERMINAL_STATUSES

    def test_non_terminal_statuses_not_in_set(self):
        for status in ["uploaded", "validated", "queued", "running"]:
            assert status not in TERMINAL_STATUSES

    def test_terminal_statuses_is_frozenset(self):
        assert isinstance(TERMINAL_STATUSES, frozenset)

    def test_exactly_three_terminal_statuses(self):
        assert len(TERMINAL_STATUSES) == 3


class TestAllowedTransitions:

    def test_all_statuses_have_entries(self):
        for status in ["uploaded", "validated", "queued", "running",
                       "completed", "failed", "cancelled"]:
            assert status in ALLOWED_TRANSITIONS

    
    def test_uploaded_to_validated(self):
        assert "validated" in ALLOWED_TRANSITIONS["uploaded"]

    def test_validated_to_queued(self):
        assert "queued" in ALLOWED_TRANSITIONS["validated"]

    def test_queued_to_running(self):
        assert "running" in ALLOWED_TRANSITIONS["queued"]

    def test_running_to_completed(self):
        assert "completed" in ALLOWED_TRANSITIONS["running"]

    

    @pytest.mark.parametrize("status", ["uploaded", "validated", "queued", "running"])
    def test_cancelled_always_allowed(self, status):
        assert "cancelled" in ALLOWED_TRANSITIONS[status]

    @pytest.mark.parametrize("status", ["uploaded", "validated", "queued", "running"])
    def test_failed_always_allowed(self, status):
        assert "failed" in ALLOWED_TRANSITIONS[status]


    @pytest.mark.parametrize("terminal", ["completed", "failed", "cancelled"])
    def test_terminal_has_no_transitions(self, terminal):
        assert len(ALLOWED_TRANSITIONS[terminal]) == 0


    def test_uploaded_cannot_jump_to_running(self):
        assert "running" not in ALLOWED_TRANSITIONS["uploaded"]

    def test_uploaded_cannot_jump_to_completed(self):
        assert "completed" not in ALLOWED_TRANSITIONS["uploaded"]

    def test_validated_cannot_jump_to_completed(self):
        assert "completed" not in ALLOWED_TRANSITIONS["validated"]

    def test_queued_cannot_jump_to_completed(self):
        assert "completed" not in ALLOWED_TRANSITIONS["queued"]


    def test_running_cannot_go_back_to_queued(self):
        assert "queued" not in ALLOWED_TRANSITIONS["running"]

    def test_completed_cannot_go_back_to_running(self):
        assert "running" not in ALLOWED_TRANSITIONS["completed"]


    def test_transition_is_allowed_helper(self):
        """Pattern for checking transitions in application code."""
        def is_allowed(current: str, target: str) -> bool:
            return target in ALLOWED_TRANSITIONS.get(current, frozenset())

        assert is_allowed("uploaded",  "validated") is True
        assert is_allowed("uploaded",  "running")   is False
        assert is_allowed("completed", "running")   is False
        assert is_allowed("running",   "completed") is True



class TestJobCreate:

    def test_happy_path_minimal(self):
        job = V1.JobCreate(
            user_id=VALID_UUID,
            idempotency_key=VALID_IKEY,
            filename="dataset.csv",
        )
        assert job.user_id         == VALID_UUID
        assert job.idempotency_key == VALID_IKEY
        assert job.filename        == "dataset.csv"
        assert job.target_column   is None
        assert job.problem_type    is None

    def test_all_fields_provided(self):
        job = V1.JobCreate(
            user_id=VALID_UUID,
            idempotency_key=VALID_IKEY,
            filename="train.parquet",
            target_column="churn",
            problem_type="classification",
        )
        assert job.target_column == "churn"
        assert job.problem_type  == "classification"

    def test_all_problem_types(self):
        for pt in ["regression", "classification", "forecasting"]:
            job = V1.JobCreate(
                user_id=VALID_UUID,
                idempotency_key=VALID_IKEY,
                filename="data.csv",
                problem_type=pt,
            )
            assert job.problem_type == pt

    def test_invalid_user_id_raises(self):
        with pytest.raises(ValidationError, match="string_pattern_mismatch"):
            V1.JobCreate(
                user_id="not-a-valid-uuid",
                idempotency_key=VALID_IKEY,
                filename="data.csv",
            )

    def test_invalid_idempotency_key_raises(self):
        # "short" is 5 chars — fails the {16,128} regex, not a min_length field.
        with pytest.raises(ValidationError, match="string_pattern_mismatch"):
            V1.JobCreate(
                user_id=VALID_UUID,
                idempotency_key="short",
                filename="data.csv",
            )



class TestJobStatusResponse:

    def _make(self, **kw):
        defaults = dict(
            job_id=VALID_IKEY, status="running", progress=50, updated_at=UTC_DT
        )
        defaults.update(kw)
        return V1.JobStatusResponse(**defaults)

    def test_happy_path(self):
        resp = self._make()
        assert resp.status   == "running"
        assert resp.progress == 50

    @pytest.mark.parametrize("status", [
        "uploaded", "validated", "queued", "running",
        "completed", "failed", "cancelled",
    ])
    def test_all_statuses_accepted(self, status):
        assert self._make(status=status).status == status

    @pytest.mark.parametrize("progress", [0, 50, 100])
    def test_progress_boundaries(self, progress):
        assert self._make(progress=progress).progress == progress

    def test_progress_negative_raises(self):
        with pytest.raises(ValidationError, match="greater_than_equal"):
            self._make(progress=-1)

    def test_progress_over_100_raises(self):
        with pytest.raises(ValidationError, match="less_than_equal"):
            self._make(progress=101)

    def test_updated_at_stored(self):
        resp = self._make()
        assert resp.updated_at == UTC_DT



class TestJobResultResponse:

    def _make(self, **kw):
        defaults = dict(
            job_id=VALID_IKEY,
            status="completed",
            model_version=VALID_VERSION,
            performance={"auc": 0.91, "f1": 0.87},
        )
        defaults.update(kw)
        return V1.JobResultResponse(**defaults)

    def test_happy_path(self):
        resp = self._make()
        assert resp.status         == "completed"
        assert resp.model_version  == VALID_VERSION
        assert resp.performance    == {"auc": 0.91, "f1": 0.87}
        assert resp.insight_summary is None

    def test_insight_summary_optional(self):
        resp = self._make(insight_summary="Strong seasonal pattern detected.")
        assert resp.insight_summary == "Strong seasonal pattern detected."

    def test_status_must_be_completed(self):
        with pytest.raises(ValidationError, match="literal_error"):
            self._make(status="running")

    def test_invalid_model_version_raises(self):
        with pytest.raises(ValidationError, match="string_pattern_mismatch"):
            self._make(model_version="v1.0.0")

    def test_performance_accepts_any_dict(self):
        resp = self._make(performance={"custom_metric": 0.5, "nested": {"a": 1}})
        assert resp.performance["custom_metric"] == 0.5



class TestFeedbackRequest:
    """
    NOTE: FeedbackRequest is defined TWICE in V1.
    Python class bodies execute top-to-bottom — the second definition
    (model_name, prediction, actual, metadata) overwrites the first
    (job_id, rating, comment). Tests target the second definition only.
    """

    def test_second_definition_wins(self):
        fields = list(V1.FeedbackRequest.model_fields.keys())
        assert "model_name" in fields
        assert "prediction" in fields
        assert "actual"     in fields
        # First definition fields must NOT be present
        assert "job_id"  not in fields
        assert "rating"  not in fields
        assert "comment" not in fields

    def test_happy_path(self):
        fb = V1.FeedbackRequest(
            model_name="churn_v2",
            prediction=0.87,
            actual=1.0,
        )
        assert fb.model_name == "churn_v2"
        assert fb.prediction == 0.87
        assert fb.actual     == 1.0
        assert fb.metadata   is None

    def test_metadata_optional(self):
        fb = V1.FeedbackRequest(
            model_name="m",
            prediction=0.5,
            actual=0.0,
            metadata={"experiment": "exp-001", "cohort": "A"},
        )
        assert fb.metadata["cohort"] == "A"

    def test_missing_model_name_raises(self):
        with pytest.raises(ValidationError):
            V1.FeedbackRequest(prediction=0.5, actual=1.0)

    def test_missing_prediction_raises(self):
        with pytest.raises(ValidationError):
            V1.FeedbackRequest(model_name="m", actual=1.0)

    def test_missing_actual_raises(self):
        with pytest.raises(ValidationError):
            V1.FeedbackRequest(model_name="m", prediction=0.5)



class TestFeedbackResponse:
    """
    NOTE: FeedbackResponse is also defined twice in V1.
    The second definition (feedback_id, model_name, errors, recorded_at, ...)
    overwrites the first (status, message).
    """

    def test_second_definition_wins(self):
        fields = list(V1.FeedbackResponse.model_fields.keys())
        assert "feedback_id"     in fields
        assert "model_name"      in fields
        assert "prediction"      in fields
        assert "actual"          in fields
        assert "absolute_error"  in fields
        assert "squared_error"   in fields
        assert "recorded_at"     in fields
        # First definition fields must NOT be present
        assert "status"   not in fields
        assert "message"  not in fields

    def test_happy_path(self):
        resp = V1.FeedbackResponse(
            feedback_id="fb-001",
            model_name="churn_v2",
            prediction=0.87,
            actual=1.0,
            absolute_error=0.13,
            squared_error=0.0169,
            relative_error=0.13,
            recorded_at=UTC_DT,
            metadata={"source": "api"},
        )
        assert resp.feedback_id    == "fb-001"
        assert resp.absolute_error == 0.13
        assert resp.recorded_at    == UTC_DT

    def test_relative_error_optional(self):
        resp = V1.FeedbackResponse(
            feedback_id="fb-002",
            model_name="m",
            prediction=0.0,
            actual=0.0,
            absolute_error=0.0,
            squared_error=0.0,
            relative_error=None,   # division by zero case
            recorded_at=UTC_DT,
            metadata={},
        )
        assert resp.relative_error is None



class TestValidateFilename:
    """
    validate_filename is defined as a plain def inside V1 (no @staticmethod).
    It is called as V1.validate_filename(v) — the unbound class-level function.
    NOTE: V1 also has a @staticmethod validate_filename stub (line 116) that
    does `pass`. Python processes class body top-to-bottom, so the plain-def
    version (line 138) overwrites the staticmethod stub.
    """

    

    @pytest.mark.parametrize("filename", [
        "data.csv",
        "my-file.json",
        "my_file.parquet",
        "my file.csv",
        "Train_Data-v2.xlsx",
        "a.py",
        "file.tar",
    ])
    def test_valid_filenames(self, filename):
        assert V1.validate_filename(filename) == filename

    def test_strips_surrounding_whitespace(self):
        assert V1.validate_filename("  data.csv  ") == "data.csv"

    def test_exactly_255_chars_allowed(self):
        name = "a" * 251 + ".csv"      # 255 chars total
        assert len(name) == 255
        assert V1.validate_filename(name) == name

    def test_extension_up_to_10_chars(self):
        name = "file." + "x" * 10     # 10-char extension
        assert V1.validate_filename(name) == name


    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            V1.validate_filename("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            V1.validate_filename("   ")

    def test_no_extension_raises(self):
        with pytest.raises(ValueError, match="contains disallowed characters"):
            V1.validate_filename("nodotfile")

    def test_extension_too_long_raises(self):
        """Extension longer than 10 chars is rejected."""
        with pytest.raises(ValueError, match="contains disallowed characters"):
            V1.validate_filename("file." + "x" * 11)

    def test_256_chars_raises(self):
        name = "a" * 252 + ".csv"      # 256 chars total
        assert len(name) == 256
        with pytest.raises(ValueError, match="must not exceed 255"):
            V1.validate_filename(name)

    def test_path_traversal_raises(self):
        with pytest.raises(ValueError, match="contains disallowed characters"):
            V1.validate_filename("../etc/passwd.txt")

    def test_leading_dot_raises(self):
        with pytest.raises(ValueError, match="contains disallowed characters"):
            V1.validate_filename(".hidden.csv")

    def test_double_dot_raises(self):
        with pytest.raises(ValueError, match="contains disallowed characters"):
            V1.validate_filename("file..csv")

    def test_return_value_is_stripped(self):
        """After stripping, the stripped value is returned."""
        result = V1.validate_filename("  report.pdf  ")
        assert result == "report.pdf"
        assert not result.startswith(" ")



class TestValidateColumnName:


    @pytest.mark.parametrize("name", [
        "price",
        "target_column",
        "col1",
        "A",
        "_private",
        "Col_Name_123",
        "x" * 128,            # exactly at max length
    ])
    def test_valid_column_names(self, name):
        assert V1.validate_column_name(name) == name

    def test_none_returns_none(self):
        assert V1.validate_column_name(None) is None

    def test_strips_surrounding_whitespace(self):
        assert V1.validate_column_name("  price  ") == "price"

    def test_exactly_128_chars_allowed(self):
        name = "x" * 128
        assert V1.validate_column_name(name) == name


    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="must not be an empty string"):
            V1.validate_column_name("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="must not be an empty string"):
            V1.validate_column_name("   ")

    def test_hyphen_raises(self):
        with pytest.raises(ValueError, match="must contain only word characters"):
            V1.validate_column_name("col-name")

    def test_space_raises(self):
        with pytest.raises(ValueError, match="must contain only word characters"):
            V1.validate_column_name("col name")

    def test_dot_raises(self):
        with pytest.raises(ValueError, match="must contain only word characters"):
            V1.validate_column_name("col.name")

    def test_at_sign_raises(self):
        with pytest.raises(ValueError, match="must contain only word characters"):
            V1.validate_column_name("col@name")

    def test_129_chars_raises(self):
        with pytest.raises(ValueError, match="must not exceed 128"):
            V1.validate_column_name("x" * 129)



class TestEnsureUtc:

    def test_naive_datetime_raises(self):
        naive = datetime(2024, 1, 1, 12, 0, 0)
        assert naive.tzinfo is None
        with pytest.raises(ValueError, match="timezone-aware"):
            V1.ensure_utc(naive)

    def test_utc_datetime_passthrough(self):
        utc = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = V1.ensure_utc(utc)
        assert result.tzinfo == timezone.utc
        assert result.hour   == 10
        assert result.minute == 30

    def test_non_utc_timezone_normalised_to_utc(self):
        """14:00 UTC+2 → 12:00 UTC."""
        plus2 = timezone(timedelta(hours=2))
        dt    = datetime(2024, 1, 1, 14, 0, 0, tzinfo=plus2)
        result = V1.ensure_utc(dt)
        assert result.tzinfo == timezone.utc
        assert result.hour   == 12

    def test_negative_offset_normalised(self):
        """08:00 UTC-5 → 13:00 UTC."""
        minus5 = timezone(timedelta(hours=-5))
        dt     = datetime(2024, 1, 1, 8, 0, 0, tzinfo=minus5)
        result = V1.ensure_utc(dt)
        assert result.tzinfo == timezone.utc
        assert result.hour   == 13

    def test_result_is_new_object_not_same(self):
        """astimezone always returns a new datetime object."""
        plus2 = timezone(timedelta(hours=2))
        dt    = datetime(2024, 1, 1, 14, 0, 0, tzinfo=plus2)
        result = V1.ensure_utc(dt)
        assert result is not dt

    def test_utc_offset_of_result_is_zero(self):
        plus2 = timezone(timedelta(hours=2))
        dt    = datetime(2024, 3, 10, 9, 0, 0, tzinfo=plus2)
        result = V1.ensure_utc(dt)
        assert result.utcoffset().total_seconds() == 0



class TestSchemaBugs:
    """
    Documents two known issues in job_schema.py so regressions are visible.
    These tests do NOT fix the bugs — they pin the current behaviour.
    """

    def test_feedback_request_second_definition_overwrites_first(self):
        """
        BUG: FeedbackRequest is defined twice inside V1.
        The second definition (model_name, prediction, actual) silently
        overwrites the first (job_id, rating, comment).
        The first definition is unreachable.
        """
        fields = set(V1.FeedbackRequest.model_fields.keys())
        # Second definition fields are present
        assert fields >= {"model_name", "prediction", "actual"}
        # First definition fields are gone
        assert "job_id"  not in fields
        assert "rating"  not in fields
        assert "comment" not in fields

    def test_static_validate_filename_stub_returns_none(self):
        """
        BUG: @staticmethod validate_filename at line 116 contains `pass`
        and returns None. The real implementation is the plain def at line 138.
        In Python the plain def (line 138) overwrites the staticmethod,
        so V1.validate_filename currently calls the real implementation.
        This test documents that if the order were reversed, callers would
        silently receive None.
        """
        # Current behaviour: real implementation is active
        result = V1.validate_filename("report.csv")
        assert result == "report.csv", (
            "validate_filename must return the stripped filename, not None. "
            "If this fails, the stub has been reactivated."
        )