"""
tests/test_routes_feedback.py

Full test suite for app/api/routes_feedback.py

Coverage
────────
POST /feedback/
    ✓ happy path — valid payload → 201 + FeedbackResponse
    ✓ response contains all required fields
    ✓ feedback_id is a valid UUID string
    ✓ model_name echoed in response
    ✓ prediction echoed in response
    ✓ actual echoed in response
    ✓ metadata echoed in response
    ✓ omitted metadata defaults to {} in response
    ✓ recorded_at is present and parseable as datetime
    ✓ FeedbackService.store_feedback called with correct args
    ✓ model_name  → model_id      (field name mapping)
    ✓ metadata    → input_data    (field name mapping)
    ✓ actual      → actual_value  (field name mapping)
    ✓ None metadata passed as {} to store_feedback (not None)
    ✓ FeedbackService called exactly once per request

_compute_metrics (unit tests — no HTTP)
    ✓ absolute_error = |actual - prediction|
    ✓ squared_error  = (actual - prediction)²
    ✓ relative_error = |actual - prediction| / |actual|
    ✓ relative_error is None when actual == 0
    ✓ prediction > actual (negative error handled correctly)
    ✓ prediction == actual → all errors are 0
    ✓ large values
    ✓ negative actual value

Error metrics in response
    ✓ absolute_error correct value
    ✓ squared_error correct value
    ✓ relative_error correct value
    ✓ relative_error is None when actual == 0

Schema validation (422)
    ✓ missing model_name → 422
    ✓ missing prediction → 422
    ✓ missing actual → 422
    ✓ empty model_name string → 422
    ✓ model_name exceeds 128 chars → 422
    ✓ FeedbackService not called on schema failure

Error handling (500)
    ✓ FeedbackService.store_feedback raises → 500
    ✓ 500 detail does not expose internal exception message
    ✓ ErrorLogger.log_error called with correct component
    ✓ ErrorLogger.log_error called with context containing model_name
    ✓ ErrorLogger.log_error called with context containing feedback_id
    ✓ ErrorLogger NOT called on successful request

GET /feedback/{model_name}
    ✓ returns 200 with list of records
    ✓ empty list returned (not 404) when no records exist
    ✓ FeedbackService.get_feedback called with model_id = model_name
    ✓ records returned unchanged from FeedbackService
    ✓ FeedbackService raises → 500
    ✓ ErrorLogger called on get_feedback failure
    ✓ ErrorLogger NOT called on successful get

Run
───
    pip install pytest pytest-asyncio httpx fastapi pydantic
    pytest tests/test_routes_feedback.py -v
"""

from __future__ import annotations

import math
from datetime import datetime
from unittest.mock import MagicMock, patch, call

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes_feedback import router, _compute_metrics



@pytest.fixture
def app():
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def mock_feedback_service():
    """
    Patches FeedbackService at the routes_feedback module level.
    store_feedback → returns None (classmethod, no return value)
    get_feedback   → returns empty list by default
    """
    with patch("app.api.routes_feedback.FeedbackService") as mock:
        mock.store_feedback.return_value = None
        mock.get_feedback.return_value   = []
        yield mock


@pytest.fixture
def mock_error_logger():
    """Patches ErrorLogger so no real logging happens in tests."""
    with patch("app.api.routes_feedback.ErrorLogger") as mock:
        yield mock


@pytest.fixture
def client(app, mock_feedback_service, mock_error_logger):
    """TestClient with both FeedbackService and ErrorLogger fully mocked."""
    with TestClient(app) as c:
        yield c



def _payload(**overrides) -> dict:
    """Minimal valid POST /feedback/ body."""
    base = {
        "model_name": "sales_forecast_model",
        "prediction": 120.0,
        "actual":     150.0,
    }
    base.update(overrides)
    return base



class TestSubmitFeedbackHappyPath:

    def test_valid_request_returns_201(self, client):
        resp = client.post("/feedback/", json=_payload())
        assert resp.status_code == 201

    def test_response_contains_all_required_fields(self, client):
        body = client.post("/feedback/", json=_payload()).json()
        required = {
            "feedback_id", "model_name", "prediction", "actual",
            "absolute_error", "squared_error", "relative_error",
            "recorded_at", "metadata",
        }
        assert required.issubset(body.keys())

    def test_feedback_id_is_valid_uuid(self, client):
        """feedback_id must be a UUID string generated fresh per request."""
        import uuid
        body = client.post("/feedback/", json=_payload()).json()
        uuid.UUID(body["feedback_id"])   # raises if not a valid UUID

    def test_two_requests_get_different_feedback_ids(self, client):
        id1 = client.post("/feedback/", json=_payload()).json()["feedback_id"]
        id2 = client.post("/feedback/", json=_payload()).json()["feedback_id"]
        assert id1 != id2

    def test_model_name_echoed_in_response(self, client):
        body = client.post(
            "/feedback/", json=_payload(model_name="churn_model")
        ).json()
        assert body["model_name"] == "churn_model"

    def test_prediction_echoed_in_response(self, client):
        body = client.post(
            "/feedback/", json=_payload(prediction=99.5)
        ).json()
        assert body["prediction"] == 99.5

    def test_actual_echoed_in_response(self, client):
        body = client.post(
            "/feedback/", json=_payload(actual=110.0)
        ).json()
        assert body["actual"] == 110.0

    def test_metadata_echoed_in_response(self, client):
        meta = {"job_id": "abc-123", "region": "EU"}
        body = client.post(
            "/feedback/", json=_payload(metadata=meta)
        ).json()
        assert body["metadata"] == meta

    def test_omitted_metadata_defaults_to_empty_dict(self, client):
        """metadata is Optional — omitting it must produce {} in the response."""
        body = client.post("/feedback/", json=_payload()).json()
        assert body["metadata"] == {}

    def test_recorded_at_is_parseable_datetime(self, client):
        body = client.post("/feedback/", json=_payload()).json()
        assert "recorded_at" in body
        # Must be parseable — raises ValueError if not
        datetime.fromisoformat(body["recorded_at"].replace("Z", "+00:00"))


class TestFeedbackServiceCalled:

    def test_store_feedback_called_once(self, client, mock_feedback_service):
        client.post("/feedback/", json=_payload())
        mock_feedback_service.store_feedback.assert_called_once()

    def test_model_name_passed_as_model_id(self, client, mock_feedback_service):
        """
        FeedbackService.store_feedback uses 'model_id', not 'model_name'.
        The route must map payload.model_name → model_id.
        """
        client.post("/feedback/", json=_payload(model_name="my_model"))
        kwargs = mock_feedback_service.store_feedback.call_args.kwargs
        assert kwargs["model_id"] == "my_model"

    def test_actual_passed_as_actual_value(self, client, mock_feedback_service):
        """
        FeedbackService.store_feedback uses 'actual_value', not 'actual'.
        The route must map payload.actual → actual_value.
        """
        client.post("/feedback/", json=_payload(actual=200.0))
        kwargs = mock_feedback_service.store_feedback.call_args.kwargs
        assert kwargs["actual_value"] == 200.0

    def test_metadata_passed_as_input_data(self, client, mock_feedback_service):
        """
        FeedbackService.store_feedback uses 'input_data', not 'metadata'.
        The route must map payload.metadata → input_data.
        """
        meta = {"job_id": "xyz"}
        client.post("/feedback/", json=_payload(metadata=meta))
        kwargs = mock_feedback_service.store_feedback.call_args.kwargs
        assert kwargs["input_data"] == meta

    def test_none_metadata_passed_as_empty_dict(self, client, mock_feedback_service):
        """
        When metadata is None, store_feedback must receive {} not None.
        Postgres cannot store None as JSON input_data.
        """
        client.post("/feedback/", json=_payload())   # no metadata key
        kwargs = mock_feedback_service.store_feedback.call_args.kwargs
        assert kwargs["input_data"] == {}

    def test_prediction_passed_correctly(self, client, mock_feedback_service):
        client.post("/feedback/", json=_payload(prediction=42.0))
        kwargs = mock_feedback_service.store_feedback.call_args.kwargs
        assert kwargs["prediction"] == 42.0

    def test_store_feedback_not_called_on_schema_failure(
        self, client, mock_feedback_service
    ):
        """Schema validation fails → FeedbackService must never be called."""
        client.post("/feedback/", json={"model_name": "m"})  # missing prediction + actual
        mock_feedback_service.store_feedback.assert_not_called()



class TestComputeMetrics:
    """
    Tests for _compute_metrics(prediction, actual) directly.
    No mocking needed — pure function.
    """

    def test_absolute_error_basic(self):
        result = _compute_metrics(prediction=120.0, actual=150.0)
        assert result["absolute_error"] == pytest.approx(30.0)

    def test_squared_error_basic(self):
        result = _compute_metrics(prediction=120.0, actual=150.0)
        assert result["squared_error"] == pytest.approx(900.0)

    def test_relative_error_basic(self):
        result = _compute_metrics(prediction=120.0, actual=150.0)
        assert result["relative_error"] == pytest.approx(30.0 / 150.0)

    def test_relative_error_is_none_when_actual_is_zero(self):
        result = _compute_metrics(prediction=10.0, actual=0.0)
        assert result["relative_error"] is None

    def test_prediction_greater_than_actual(self):
        """Prediction over-estimates — absolute error still positive."""
        result = _compute_metrics(prediction=200.0, actual=100.0)
        assert result["absolute_error"] == pytest.approx(100.0)
        assert result["squared_error"]  == pytest.approx(10000.0)

    def test_prediction_equals_actual_all_zeros(self):
        result = _compute_metrics(prediction=50.0, actual=50.0)
        assert result["absolute_error"] == pytest.approx(0.0)
        assert result["squared_error"]  == pytest.approx(0.0)
        assert result["relative_error"] == pytest.approx(0.0)

    def test_negative_actual_value(self):
        """absolute_error uses abs(actual) for relative_error denominator."""
        result = _compute_metrics(prediction=0.0, actual=-10.0)
        assert result["absolute_error"] == pytest.approx(10.0)
        assert result["relative_error"] == pytest.approx(10.0 / 10.0)

    def test_large_values(self):
        result = _compute_metrics(prediction=1_000_000.0, actual=1_000_001.0)
        assert result["absolute_error"] == pytest.approx(1.0)
        assert result["squared_error"]  == pytest.approx(1.0)

    def test_fractional_values(self):
        result = _compute_metrics(prediction=0.1, actual=0.3)
        assert result["absolute_error"] == pytest.approx(0.2, rel=1e-5)



class TestErrorMetricsInResponse:

    def test_absolute_error_in_response(self, client):
        body = client.post(
            "/feedback/", json=_payload(prediction=120.0, actual=150.0)
        ).json()
        assert body["absolute_error"] == pytest.approx(30.0)

    def test_squared_error_in_response(self, client):
        body = client.post(
            "/feedback/", json=_payload(prediction=120.0, actual=150.0)
        ).json()
        assert body["squared_error"] == pytest.approx(900.0)

    def test_relative_error_in_response(self, client):
        body = client.post(
            "/feedback/", json=_payload(prediction=120.0, actual=150.0)
        ).json()
        assert body["relative_error"] == pytest.approx(30.0 / 150.0)

    def test_relative_error_is_null_when_actual_is_zero(self, client):
        body = client.post(
            "/feedback/", json=_payload(prediction=10.0, actual=0.0)
        ).json()
        assert body["relative_error"] is None

    def test_all_errors_zero_when_prediction_equals_actual(self, client):
        body = client.post(
            "/feedback/", json=_payload(prediction=50.0, actual=50.0)
        ).json()
        assert body["absolute_error"] == pytest.approx(0.0)
        assert body["squared_error"]  == pytest.approx(0.0)
        assert body["relative_error"] == pytest.approx(0.0)



class TestSchemaValidation:

    def test_missing_model_name_returns_422(self, client, mock_feedback_service):
        resp = client.post(
            "/feedback/", json={"prediction": 100.0, "actual": 110.0}
        )
        assert resp.status_code == 422
        mock_feedback_service.store_feedback.assert_not_called()

    def test_missing_prediction_returns_422(self, client, mock_feedback_service):
        resp = client.post(
            "/feedback/", json={"model_name": "m", "actual": 110.0}
        )
        assert resp.status_code == 422
        mock_feedback_service.store_feedback.assert_not_called()

    def test_missing_actual_returns_422(self, client, mock_feedback_service):
        resp = client.post(
            "/feedback/", json={"model_name": "m", "prediction": 100.0}
        )
        assert resp.status_code == 422
        mock_feedback_service.store_feedback.assert_not_called()

    def test_empty_model_name_returns_422(self, client, mock_feedback_service):
        """model_name has min_length=1."""
        resp = client.post("/feedback/", json=_payload(model_name=""))
        assert resp.status_code == 422
        mock_feedback_service.store_feedback.assert_not_called()

    def test_model_name_over_128_chars_returns_422(self, client, mock_feedback_service):
        """model_name has max_length=128."""
        resp = client.post("/feedback/", json=_payload(model_name="x" * 129))
        assert resp.status_code == 422
        mock_feedback_service.store_feedback.assert_not_called()

    def test_model_name_exactly_128_chars_accepted(self, client):
        resp = client.post("/feedback/", json=_payload(model_name="x" * 128))
        assert resp.status_code == 201

    def test_prediction_as_string_returns_422(self, client, mock_feedback_service):
        """prediction must be a float — string should fail Pydantic coercion."""
        resp = client.post(
            "/feedback/", json={"model_name": "m", "prediction": "bad", "actual": 1.0}
        )
        assert resp.status_code == 422
        mock_feedback_service.store_feedback.assert_not_called()



class TestSubmitFeedbackErrorHandling:

    def test_store_feedback_exception_returns_500(
        self, client, mock_feedback_service
    ):
        mock_feedback_service.store_feedback.side_effect = RuntimeError(
            "postgres connection refused"
        )
        resp = client.post("/feedback/", json=_payload())
        assert resp.status_code == 500

    def test_500_detail_is_generic(self, client, mock_feedback_service):
        """Internal error message must NOT leak to the client."""
        mock_feedback_service.store_feedback.side_effect = RuntimeError(
            "password: hunter2"
        )
        body = client.post("/feedback/", json=_payload()).json()
        assert "hunter2"             not in body["detail"]
        assert "postgres"            not in body["detail"].lower()
        assert "Failed to record"    in body["detail"]

    def test_error_logger_called_on_exception(
        self, client, mock_feedback_service, mock_error_logger
    ):
        mock_feedback_service.store_feedback.side_effect = RuntimeError("db error")
        client.post("/feedback/", json=_payload())
        mock_error_logger.log_error.assert_called_once()

    def test_error_logger_component_is_correct(
        self, client, mock_feedback_service, mock_error_logger
    ):
        mock_feedback_service.store_feedback.side_effect = RuntimeError("db error")
        client.post("/feedback/", json=_payload())
        kwargs = mock_error_logger.log_error.call_args.kwargs
        assert kwargs["component"] == "routes_feedback/submit"

    def test_error_logger_context_contains_model_name(
        self, client, mock_feedback_service, mock_error_logger
    ):
        mock_feedback_service.store_feedback.side_effect = RuntimeError("db error")
        client.post("/feedback/", json=_payload(model_name="my_model"))
        kwargs = mock_error_logger.log_error.call_args.kwargs
        assert kwargs["context"]["model_name"] == "my_model"

    def test_error_logger_context_contains_prediction_and_actual(
        self, client, mock_feedback_service, mock_error_logger
    ):
        mock_feedback_service.store_feedback.side_effect = RuntimeError("db error")
        client.post("/feedback/", json=_payload(prediction=10.0, actual=20.0))
        kwargs = mock_error_logger.log_error.call_args.kwargs
        assert kwargs["context"]["prediction"] == 10.0
        assert kwargs["context"]["actual"]     == 20.0

    def test_error_logger_context_contains_feedback_id(
        self, client, mock_feedback_service, mock_error_logger
    ):
        """feedback_id must be in the error context so failures are traceable."""
        mock_feedback_service.store_feedback.side_effect = RuntimeError("db error")
        client.post("/feedback/", json=_payload())
        kwargs = mock_error_logger.log_error.call_args.kwargs
        assert "feedback_id" in kwargs["context"]

    def test_error_logger_receives_the_exception_object(
        self, client, mock_feedback_service, mock_error_logger
    ):
        exc = RuntimeError("specific error")
        mock_feedback_service.store_feedback.side_effect = exc
        client.post("/feedback/", json=_payload())
        kwargs = mock_error_logger.log_error.call_args.kwargs
        assert kwargs["error"] is exc

    def test_error_logger_not_called_on_success(
        self, client, mock_feedback_service, mock_error_logger
    ):
        """ErrorLogger must NOT be called when the request succeeds."""
        client.post("/feedback/", json=_payload())
        mock_error_logger.log_error.assert_not_called()


class TestGetFeedback:

    def test_returns_200(self, client, mock_feedback_service):
        resp = client.get("/feedback/my_model")
        assert resp.status_code == 200

    def test_empty_list_returned_when_no_records(self, client, mock_feedback_service):
        """No records is a valid state — must return [] not 404."""
        mock_feedback_service.get_feedback.return_value = []
        body = client.get("/feedback/my_model").json()
        assert body == []

    def test_records_returned_unchanged(self, client, mock_feedback_service):
        """Whatever FeedbackService returns must be passed through unchanged."""
        records = [
            {"model_id": "m", "prediction": 100.0, "actual_value": 110.0},
            {"model_id": "m", "prediction": 90.0,  "actual_value": 95.0},
        ]
        mock_feedback_service.get_feedback.return_value = records
        body = client.get("/feedback/my_model").json()
        assert body == records

    def test_get_feedback_called_with_model_id(self, client, mock_feedback_service):
        """model_name path param must be passed as model_id to FeedbackService."""
        client.get("/feedback/churn_predictor")
        mock_feedback_service.get_feedback.assert_called_once_with(
            model_id="churn_predictor"
        )

    def test_get_feedback_exception_returns_500(
        self, client, mock_feedback_service
    ):
        mock_feedback_service.get_feedback.side_effect = RuntimeError(
            "postgres timeout"
        )
        resp = client.get("/feedback/my_model")
        assert resp.status_code == 500

    def test_500_detail_contains_model_name(
        self, client, mock_feedback_service
    ):
        mock_feedback_service.get_feedback.side_effect = RuntimeError("timeout")
        body = client.get("/feedback/my_model").json()
        assert "my_model" in body["detail"]

    def test_error_logger_called_on_get_failure(
        self, client, mock_feedback_service, mock_error_logger
    ):
        mock_feedback_service.get_feedback.side_effect = RuntimeError("timeout")
        client.get("/feedback/my_model")
        mock_error_logger.log_error.assert_called_once()

    def test_error_logger_component_on_get_failure(
        self, client, mock_feedback_service, mock_error_logger
    ):
        mock_feedback_service.get_feedback.side_effect = RuntimeError("timeout")
        client.get("/feedback/my_model")
        kwargs = mock_error_logger.log_error.call_args.kwargs
        assert kwargs["component"] == "routes_feedback/get_feedback"

    def test_error_logger_context_contains_model_name_on_get_failure(
        self, client, mock_feedback_service, mock_error_logger
    ):
        mock_feedback_service.get_feedback.side_effect = RuntimeError("timeout")
        client.get("/feedback/churn_model")
        kwargs = mock_error_logger.log_error.call_args.kwargs
        assert kwargs["context"]["model_name"] == "churn_model"

    def test_error_logger_not_called_on_successful_get(
        self, client, mock_feedback_service, mock_error_logger
    ):
        client.get("/feedback/my_model")
        mock_error_logger.log_error.assert_not_called()