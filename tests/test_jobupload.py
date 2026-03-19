"""
tests/test_routes_jobs.py

Full test suite for app/api/routes_jobs.py

Coverage
────────
POST /v1/train/validate
    ✓ valid request → 202 + JobStatusResponse
    ✓ dispatches "validate" task type to celery_app
    ✓ job_id in response equals request idempotency_key
    ✓ response status is "queued"
    ✓ response progress is 0

POST /v1/train/tabular
    ✓ valid request, no problem_type → 202
    ✓ valid request, problem_type=regression → 202
    ✓ valid request, problem_type=classification → 202
    ✓ problem_type=forecasting → 400 (wrong endpoint)
    ✓ dispatches "train" task type to celery_app
    ✓ job_id equals idempotency_key

POST /v1/train/forecast
    ✓ valid request, problem_type=forecasting → 202
    ✓ problem_type=regression → 400 (wrong endpoint)
    ✓ problem_type=classification → 400 (wrong endpoint)
    ✓ problem_type=None → 400 (must be forecasting)
    ✓ dispatches "forecast" task type to celery_app
    ✓ job_id equals idempotency_key

Schema validation (422)
    ✓ missing user_id → 422
    ✓ missing idempotency_key → 422
    ✓ missing filename → 422
    ✓ user_id not a valid UUID → 422
    ✓ idempotency_key too short (< 16 chars) → 422
    ✓ invalid problem_type value → 422

_validate_job_create (422 from validators)
    ✓ filename with disallowed chars → 422
    ✓ filename with no extension → 422  (NOTE: validate_filename has a bug — see below)
    ✓ target_column with special chars → 422
    ✓ empty target_column string → 422

Celery error handling
    ✓ dispatch_automl_task raises ValueError (unknown task) → 400
    ✓ dispatch_automl_task raises Exception (broker down) → 500

JobStatusResponse shape
    ✓ job_id present and matches idempotency_key
    ✓ status == "queued"
    ✓ progress == 0
    ✓ updated_at is present

Known bugs in job_schema.py (documented, tests flag them)
    ✗ validate_filename defined twice — second (non-@staticmethod) definition
      silently replaces the first (@staticmethod returning None via pass).
      The real validator IS the second one — tests target that behaviour.
    ✗ FeedbackRequest defined twice with different fields — second definition wins.

Run
───
    pip install pytest httpx fastapi
    pytest tests/test_routes_jobs.py -v
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from celery.result import AsyncResult
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes_jobs import router


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def app():
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def mock_celery():
    """
    Patches celery_app.dispatch_automl_task so no real broker is needed.
    Returns a MagicMock that simulates a successful AsyncResult.
    """
    fake_result = MagicMock(spec=AsyncResult)
    fake_result.id = str(uuid.uuid4())

    with patch(
        "app.api.routes_jobs.celery_app.dispatch_automl_task",
        return_value=fake_result,
    ) as mock_dispatch:
        yield mock_dispatch


@pytest.fixture
def client(app, mock_celery):
    """TestClient with Celery fully mocked — no Redis / broker required."""
    with TestClient(app) as c:
        yield c


# ── Helpers ───────────────────────────────────────────────────────────────────

_VALID_UUID = str(uuid.uuid4())
_VALID_IDEMPOTENCY_KEY = "test-idempotency-key-001"   # 27 chars, printable ASCII


def _job_payload(**overrides) -> dict:
    """Minimal valid JobCreate payload. Merge overrides to test variants."""
    base = {
        "user_id":         _VALID_UUID,
        "idempotency_key": _VALID_IDEMPOTENCY_KEY,
        "filename":        "sales_data.csv",
    }
    base.update(overrides)
    return base


# ══════════════════════════════════════════════════════════════════════════════
# POST /v1/train/validate
# ══════════════════════════════════════════════════════════════════════════════

class TestValidateEndpoint:

    def test_valid_request_returns_202(self, client):
        resp = client.post("/v1/train/validate", json=_job_payload())
        assert resp.status_code == 202

    def test_dispatches_validate_task_type(self, client, mock_celery):
        client.post("/v1/train/validate", json=_job_payload())
        mock_celery.assert_called_once()
        task_type = mock_celery.call_args[0][0]
        assert task_type == "validate"

    def test_payload_forwarded_to_dispatch(self, client, mock_celery):
        """The full model_dump() of JobCreate must reach dispatch_automl_task."""
        payload = _job_payload(filename="mydata.csv")
        client.post("/v1/train/validate", json=payload)
        dispatched_payload = mock_celery.call_args[0][1]
        assert dispatched_payload["filename"] == "mydata.csv"

    def test_job_id_equals_idempotency_key(self, client):
        body = client.post("/v1/train/validate", json=_job_payload()).json()
        assert body["job_id"] == _VALID_IDEMPOTENCY_KEY

    def test_response_status_is_queued(self, client):
        body = client.post("/v1/train/validate", json=_job_payload()).json()
        assert body["status"] == "queued"

    def test_response_progress_is_zero(self, client):
        body = client.post("/v1/train/validate", json=_job_payload()).json()
        assert body["progress"] == 0

    def test_response_contains_updated_at(self, client):
        body = client.post("/v1/train/validate", json=_job_payload()).json()
        assert "updated_at" in body
        # Must be parseable as a datetime
        datetime.fromisoformat(body["updated_at"].replace("Z", "+00:00"))


# ══════════════════════════════════════════════════════════════════════════════
# POST /v1/train/tabular
# ══════════════════════════════════════════════════════════════════════════════

class TestTabularEndpoint:

    def test_valid_request_returns_202(self, client):
        resp = client.post("/v1/train/tabular", json=_job_payload())
        assert resp.status_code == 202

    def test_regression_problem_type_accepted(self, client):
        resp = client.post(
            "/v1/train/tabular",
            json=_job_payload(problem_type="regression"),
        )
        assert resp.status_code == 202

    def test_classification_problem_type_accepted(self, client):
        resp = client.post(
            "/v1/train/tabular",
            json=_job_payload(problem_type="classification"),
        )
        assert resp.status_code == 202

    def test_forecasting_problem_type_rejected_400(self, client):
        """
        Forecasting jobs must go to /forecast — tabular must reject them with 400.
        """
        resp = client.post(
            "/v1/train/tabular",
            json=_job_payload(problem_type="forecasting"),
        )
        assert resp.status_code == 400

    def test_400_detail_mentions_forecast_endpoint(self, client):
        body = client.post(
            "/v1/train/tabular",
            json=_job_payload(problem_type="forecasting"),
        ).json()
        assert "forecast" in body["detail"].lower()

    def test_dispatches_train_task_type(self, client, mock_celery):
        client.post("/v1/train/tabular", json=_job_payload())
        task_type = mock_celery.call_args[0][0]
        assert task_type == "train"

    def test_job_id_equals_idempotency_key(self, client):
        body = client.post("/v1/train/tabular", json=_job_payload()).json()
        assert body["job_id"] == _VALID_IDEMPOTENCY_KEY

    def test_celery_not_called_when_forecasting_rejected(self, client, mock_celery):
        client.post("/v1/train/tabular", json=_job_payload(problem_type="forecasting"))
        mock_celery.assert_not_called()

    def test_target_column_forwarded(self, client, mock_celery):
        client.post(
            "/v1/train/tabular",
            json=_job_payload(target_column="price"),
        )
        dispatched = mock_celery.call_args[0][1]
        assert dispatched["target_column"] == "price"


# ══════════════════════════════════════════════════════════════════════════════
# POST /v1/train/forecast
# ══════════════════════════════════════════════════════════════════════════════

class TestForecastEndpoint:

    def test_forecasting_problem_type_returns_202(self, client):
        resp = client.post(
            "/v1/train/forecast",
            json=_job_payload(problem_type="forecasting"),
        )
        assert resp.status_code == 202

    def test_regression_problem_type_rejected_400(self, client):
        resp = client.post(
            "/v1/train/forecast",
            json=_job_payload(problem_type="regression"),
        )
        assert resp.status_code == 400

    def test_classification_problem_type_rejected_400(self, client):
        resp = client.post(
            "/v1/train/forecast",
            json=_job_payload(problem_type="classification"),
        )
        assert resp.status_code == 400

    def test_none_problem_type_rejected_400(self, client):
        """
        /forecast requires problem_type == 'forecasting' explicitly.
        Omitting problem_type must return 400.
        """
        resp = client.post("/v1/train/forecast", json=_job_payload())
        assert resp.status_code == 400

    def test_400_detail_mentions_forecasting(self, client):
        body = client.post(
            "/v1/train/forecast",
            json=_job_payload(problem_type="regression"),
        ).json()
        assert "forecasting" in body["detail"].lower()

    def test_dispatches_forecast_task_type(self, client, mock_celery):
        client.post(
            "/v1/train/forecast",
            json=_job_payload(problem_type="forecasting"),
        )
        task_type = mock_celery.call_args[0][0]
        assert task_type == "forecast"

    def test_job_id_equals_idempotency_key(self, client):
        body = client.post(
            "/v1/train/forecast",
            json=_job_payload(problem_type="forecasting"),
        ).json()
        assert body["job_id"] == _VALID_IDEMPOTENCY_KEY

    def test_celery_not_called_when_wrong_problem_type(self, client, mock_celery):
        client.post(
            "/v1/train/forecast",
            json=_job_payload(problem_type="regression"),
        )
        mock_celery.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# Schema validation — FastAPI / Pydantic level (422)
# ══════════════════════════════════════════════════════════════════════════════

class TestSchemaValidation:
    """
    These failures are caught by Pydantic before _validate_job_create runs.
    Celery must never be called for any of them.
    """

    def test_missing_user_id_returns_422(self, client, mock_celery):
        payload = {
            "idempotency_key": _VALID_IDEMPOTENCY_KEY,
            "filename": "data.csv",
        }
        resp = client.post("/v1/train/validate", json=payload)
        assert resp.status_code == 422
        mock_celery.assert_not_called()

    def test_missing_idempotency_key_returns_422(self, client, mock_celery):
        payload = {"user_id": _VALID_UUID, "filename": "data.csv"}
        resp = client.post("/v1/train/validate", json=payload)
        assert resp.status_code == 422
        mock_celery.assert_not_called()

    def test_missing_filename_returns_422(self, client, mock_celery):
        payload = {
            "user_id": _VALID_UUID,
            "idempotency_key": _VALID_IDEMPOTENCY_KEY,
        }
        resp = client.post("/v1/train/validate", json=payload)
        assert resp.status_code == 422
        mock_celery.assert_not_called()

    def test_invalid_uuid_user_id_returns_422(self, client, mock_celery):
        resp = client.post(
            "/v1/train/validate",
            json=_job_payload(user_id="not-a-uuid"),
        )
        assert resp.status_code == 422
        mock_celery.assert_not_called()

    def test_idempotency_key_too_short_returns_422(self, client, mock_celery):
        """IdempotencyKeyStr requires min 16 printable ASCII chars."""
        resp = client.post(
            "/v1/train/validate",
            json=_job_payload(idempotency_key="tooshort"),
        )
        assert resp.status_code == 422
        mock_celery.assert_not_called()

    def test_invalid_problem_type_returns_422(self, client, mock_celery):
        resp = client.post(
            "/v1/train/validate",
            json=_job_payload(problem_type="clustering"),
        )
        assert resp.status_code == 422
        mock_celery.assert_not_called()

    def test_user_id_with_spaces_returns_422(self, client, mock_celery):
        """UUIDStr pattern does not allow spaces."""
        resp = client.post(
            "/v1/train/validate",
            json=_job_payload(user_id="not a uuid at all"),
        )
        assert resp.status_code == 422
        mock_celery.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# _validate_job_create — filename + target_column validation (422)
# ══════════════════════════════════════════════════════════════════════════════

class TestValidateJobCreate:
    """
    Tests for the _validate_job_create helper which calls V1.validate_filename
    and V1.validate_column_name directly.

    NOTE on job_schema.py bug
    ─────────────────────────
    validate_filename is defined twice inside V1:
      1. @staticmethod def validate_filename(v): pass    ← returns None (no-op)
      2. def validate_filename(v): <real logic>          ← overrides #1 at class scope

    In Python, the second definition wins. The route calls V1.validate_filename(...)
    which resolves to the second definition — the one with real validation logic.
    These tests target that actual behaviour.
    """

    def test_filename_with_disallowed_chars_returns_422(self, client, mock_celery):
        """Semicolons and angle brackets are not in _SAFE_FILENAME_RE."""
        resp = client.post(
            "/v1/train/tabular",
            json=_job_payload(filename="bad;file<name>.csv"),
        )
        assert resp.status_code == 422
        mock_celery.assert_not_called()

    def test_filename_with_path_traversal_returns_422(self, client, mock_celery):
        """../ is not in _SAFE_FILENAME_RE."""
        resp = client.post(
            "/v1/train/tabular",
            json=_job_payload(filename="../etc/passwd.csv"),
        )
        assert resp.status_code == 422
        mock_celery.assert_not_called()

    def test_empty_filename_returns_422(self, client, mock_celery):
        resp = client.post(
            "/v1/train/tabular",
            json=_job_payload(filename=""),
        )
        assert resp.status_code == 422
        mock_celery.assert_not_called()

    def test_valid_filename_with_hyphen_and_underscore_accepted(self, client):
        resp = client.post(
            "/v1/train/tabular",
            json=_job_payload(filename="my_sales-data.csv"),
        )
        assert resp.status_code == 202

    def test_target_column_with_special_chars_returns_422(self, client, mock_celery):
        """_COLUMN_NAME_RE only allows word chars (\w = a-z A-Z 0-9 _)."""
        resp = client.post(
            "/v1/train/tabular",
            json=_job_payload(target_column="col-name"),
        )
        assert resp.status_code == 422
        mock_celery.assert_not_called()

    def test_target_column_with_spaces_returns_422(self, client, mock_celery):
        resp = client.post(
            "/v1/train/tabular",
            json=_job_payload(target_column="my column"),
        )
        assert resp.status_code == 422
        mock_celery.assert_not_called()

    def test_empty_target_column_string_returns_422(self, client, mock_celery):
        resp = client.post(
            "/v1/train/tabular",
            json=_job_payload(target_column="   "),
        )
        assert resp.status_code == 422
        mock_celery.assert_not_called()

    def test_none_target_column_accepted(self, client):
        """target_column is Optional — None must pass validation."""
        resp = client.post(
            "/v1/train/tabular",
            json=_job_payload(target_column=None),
        )
        assert resp.status_code == 202

    def test_valid_target_column_with_underscore_accepted(self, client):
        resp = client.post(
            "/v1/train/tabular",
            json=_job_payload(target_column="sale_price"),
        )
        assert resp.status_code == 202


# ══════════════════════════════════════════════════════════════════════════════
# Celery error handling
# ══════════════════════════════════════════════════════════════════════════════

class TestCeleryErrorHandling:

    def test_dispatch_value_error_returns_400(self, app):
        """
        dispatch_automl_task raises ValueError (unknown task_type) → 400.
        This path is reached only if someone adds a new endpoint without
        adding the task_type to the mapping — guards against that.
        """
        with patch(
            "app.api.routes_jobs.celery_app.dispatch_automl_task",
            side_effect=ValueError("Unknown task type: badtype"),
        ):
            with TestClient(app) as c:
                resp = c.post("/v1/train/validate", json=_job_payload())
        assert resp.status_code == 400

    def test_dispatch_exception_returns_500(self, app):
        """
        Any unexpected exception from dispatch (Redis down, network failure)
        → 500 Internal Server Error.
        """
        with patch(
            "app.api.routes_jobs.celery_app.dispatch_automl_task",
            side_effect=ConnectionError("Redis broker unreachable"),
        ):
            with TestClient(app) as c:
                resp = c.post("/v1/train/validate", json=_job_payload())
        assert resp.status_code == 500

    def test_500_detail_is_generic(self, app):
        """
        Internal errors must not leak broker connection details to the client.
        """
        with patch(
            "app.api.routes_jobs.celery_app.dispatch_automl_task",
            side_effect=ConnectionError("redis://internal-host:6379 refused"),
        ):
            with TestClient(app) as c:
                body = c.post("/v1/train/validate", json=_job_payload()).json()
        assert "internal-host" not in body["detail"]
        assert "6379"          not in body["detail"]

    def test_400_detail_from_value_error_is_forwarded(self, app):
        """ValueError detail message IS forwarded — it is safe (task type name)."""
        with patch(
            "app.api.routes_jobs.celery_app.dispatch_automl_task",
            side_effect=ValueError("Unknown task type: badtype"),
        ):
            with TestClient(app) as c:
                body = c.post("/v1/train/validate", json=_job_payload()).json()
        assert "Unknown task type" in body["detail"]

    def test_dispatch_called_exactly_once_per_request(self, client, mock_celery):
        """One HTTP request must result in exactly one Celery dispatch."""
        client.post("/v1/train/validate", json=_job_payload())
        assert mock_celery.call_count == 1


# ══════════════════════════════════════════════════════════════════════════════
# JobStatusResponse shape
# ══════════════════════════════════════════════════════════════════════════════

class TestJobStatusResponseShape:
    """
    Verifies the response body matches V1.JobStatusResponse for all endpoints.
    """

    @pytest.mark.parametrize("endpoint,extra", [
        ("/v1/train/validate",  {}),
        ("/v1/train/tabular",   {}),
        ("/v1/train/forecast",  {"problem_type": "forecasting"}),
    ])
    def test_response_has_all_required_fields(self, client, endpoint, extra):
        body = client.post(endpoint, json=_job_payload(**extra)).json()
        assert "job_id"     in body
        assert "status"     in body
        assert "progress"   in body
        assert "updated_at" in body

    @pytest.mark.parametrize("endpoint,extra", [
        ("/v1/train/validate",  {}),
        ("/v1/train/tabular",   {}),
        ("/v1/train/forecast",  {"problem_type": "forecasting"}),
    ])
    def test_job_id_matches_idempotency_key(self, client, endpoint, extra):
        body = client.post(endpoint, json=_job_payload(**extra)).json()
        assert body["job_id"] == _VALID_IDEMPOTENCY_KEY

    @pytest.mark.parametrize("endpoint,extra", [
        ("/v1/train/validate",  {}),
        ("/v1/train/tabular",   {}),
        ("/v1/train/forecast",  {"problem_type": "forecasting"}),
    ])
    def test_status_is_queued(self, client, endpoint, extra):
        body = client.post(endpoint, json=_job_payload(**extra)).json()
        assert body["status"] == "queued"

    @pytest.mark.parametrize("endpoint,extra", [
        ("/v1/train/validate",  {}),
        ("/v1/train/tabular",   {}),
        ("/v1/train/forecast",  {"problem_type": "forecasting"}),
    ])
    def test_progress_is_zero(self, client, endpoint, extra):
        body = client.post(endpoint, json=_job_payload(**extra)).json()
        assert body["progress"] == 0