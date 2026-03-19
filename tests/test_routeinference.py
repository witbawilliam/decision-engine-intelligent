"""
tests/test_routes_inference.py

Full test suite for app/api/routes_inference.py

Coverage
────────
 POST /inference/predict
    ✓ happy path — tabular pipeline, cache miss, success response
    ✓ happy path — temporal pipeline, cache miss, success response
    ✓ cache hit — service returns cached=True, status == CACHED
    ✓ schema validation — missing required field (features) → 422
    ✓ schema validation — empty features dict → 422
    ✓ schema validation — invalid model_name characters → 422
    ✓ schema validation — invalid version string → 422
    ✓ schema validation — extra unknown field → 422 (extra="forbid")
    ✓ schema validation — features exceed 500 keys → 422
    ✓ service raises ValueError (model not found) → 404
    ✓ service raises ValueError (invalid features) → 404
    ✓ service raises asyncio.TimeoutError → 504
    ✓ service raises unexpected Exception → 500
    ✓ response shape — all required fields present
    ✓ trace propagation — request trace_id forwarded into service
    ✓ pipeline_type enum — forecasting value accepted

 GET /inference/health
    ✓ both services healthy → 200 with redis + postgres True
    ✓ redis down → response reflects False
    ✓ postgres down → response reflects False

Dependencies
────────────
    pip install pytest pytest-asyncio httpx fastapi
"""

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


from app.api.routes_inference import router, get_prediction_service


from app.schemas.prediction_schema import PredictionStatus


from service.prediction_service import PredictionResponse as ServiceResponse


@pytest.fixture
def app():
    """Minimal FastAPI app with only the inference router registered."""
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def mock_service():
    """
    A MagicMock standing in for PredictionService.

    .predict()  → AsyncMock so it can be awaited in the async route.
    .health()   → MagicMock returning healthy status by default.
    """
    svc = MagicMock()
    svc.predict = AsyncMock(return_value=_make_service_response())
    svc.health  = MagicMock(return_value={"redis": True, "postgres": True})
    return svc


@pytest.fixture
def client(app, mock_service):
    """
    TestClient with the real get_prediction_service dependency
    overridden to return mock_service — no real Redis / Postgres / pipelines.
    """
    app.dependency_overrides[get_prediction_service] = lambda: mock_service
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _make_service_response(
    *,
    prediction: Any = 42.0,
    model_version: str = "1.0.0",
    cached: bool = False,
) -> ServiceResponse:
    """Build the internal dataclass that PredictionService.predict() returns."""
    return ServiceResponse(
        request_id    = str(uuid.uuid4()),
        model_name    = "sales_forecast_model",
        model_version = model_version,
        prediction    = prediction,
        latency_ms    = 12.5,
        trace_id      = uuid.uuid4().hex,
        cached        = cached,
    )


def _valid_payload(**overrides) -> Dict[str, Any]:
    """Minimal valid POST /predict body. Merge overrides to test variants."""
    base = {
        "model_name":    "sales_forecast_model",
        "pipeline_type": "tabular",
        "features":      {"age": 30, "income": 50000},
    }
    base.update(overrides)
    return base



class TestPredictHappyPath:

    def test_tabular_pipeline_returns_200(self, client, mock_service):
        """Valid tabular request → service called once → HTTP 200."""
        resp = client.post("/inference/predict", json=_valid_payload())

        assert resp.status_code == 200
        mock_service.predict.assert_awaited_once()

    def test_temporal_pipeline_accepted(self, client, mock_service):
        """pipeline_type=temporal is a valid enum value → HTTP 200."""
        resp = client.post(
            "/inference/predict",
            json=_valid_payload(pipeline_type="temporal"),
        )
        assert resp.status_code == 200

    def test_forecasting_pipeline_accepted(self, client, mock_service):
        """pipeline_type=forecasting is a valid enum value → HTTP 200."""
        resp = client.post(
            "/inference/predict",
            json=_valid_payload(pipeline_type="forecasting"),
        )
        assert resp.status_code == 200

    def test_response_contains_prediction(self, client, mock_service):
        """Response body must include the prediction value from the service."""
        mock_service.predict = AsyncMock(
            return_value=_make_service_response(prediction=99.9)
        )
        body = client.post("/inference/predict", json=_valid_payload()).json()
        assert body["prediction"] == 99.9

    def test_response_contains_model_version(self, client, mock_service):
        """model_version in response must match what the service returned."""
        mock_service.predict = AsyncMock(
            return_value=_make_service_response(model_version="2.3.1")
        )
        body = client.post("/inference/predict", json=_valid_payload()).json()
        assert body["model_version"] == "2.3.1"

    def test_status_is_success_on_cache_miss(self, client, mock_service):
        """cache miss (cached=False) → response status must be 'success'."""
        mock_service.predict = AsyncMock(
            return_value=_make_service_response(cached=False)
        )
        body = client.post("/inference/predict", json=_valid_payload()).json()
        assert body["status"] == PredictionStatus.SUCCESS.value

    def test_status_is_cached_on_cache_hit(self, client, mock_service):
        """cache hit (cached=True) → response status must be 'cached'."""
        mock_service.predict = AsyncMock(
            return_value=_make_service_response(cached=True)
        )
        body = client.post("/inference/predict", json=_valid_payload()).json()
        assert body["status"] == PredictionStatus.CACHED.value
        assert body["cached"] is True

    def test_latency_ms_is_non_negative(self, client):
        """latency_ms in response must be >= 0."""
        body = client.post("/inference/predict", json=_valid_payload()).json()
        assert body["latency_ms"] >= 0.0

    def test_response_contains_request_id(self, client):
        """request_id must be present in the response."""
        body = client.post("/inference/predict", json=_valid_payload()).json()
        assert "request_id" in body
        # Must be a valid UUID string
        uuid.UUID(body["request_id"])

    def test_response_contains_trace(self, client):
        """trace context (trace_id, span_id) must be forwarded in response."""
        body = client.post("/inference/predict", json=_valid_payload()).json()
        assert "trace" in body
        assert "trace_id" in body["trace"]
        assert "span_id"  in body["trace"]

    def test_explicit_version_accepted(self, client):
        """Semver version string is valid → no 422."""
        resp = client.post(
            "/inference/predict",
            json=_valid_payload(version="1.2.3"),
        )
        assert resp.status_code == 200

    def test_stage_alias_version_accepted(self, client):
        """Stage alias versions (production, staging, latest) are valid."""
        for alias in ("production", "staging", "latest"):
            resp = client.post(
                "/inference/predict",
                json=_valid_payload(version=alias),
            )
            assert resp.status_code == 200, f"alias '{alias}' should be accepted"



class TestPredictSchemaValidation:
    """
    FastAPI validates PredictionRequest before calling the route function.
    All these cases must return 422 — mock_service.predict must NOT be called.
    """

    def test_missing_features_returns_422(self, client, mock_service):
        payload = {"model_name": "sales_forecast_model", "pipeline_type": "tabular"}
        resp = client.post("/inference/predict", json=payload)
        assert resp.status_code == 422
        mock_service.predict.assert_not_awaited()

    def test_missing_model_name_returns_422(self, client, mock_service):
        payload = {"features": {"age": 30}, "pipeline_type": "tabular"}
        resp = client.post("/inference/predict", json=payload)
        assert resp.status_code == 422
        mock_service.predict.assert_not_awaited()

    def test_empty_features_returns_422(self, client, mock_service):
        """prediction_schema rejects empty features dict."""
        resp = client.post(
            "/inference/predict",
            json=_valid_payload(features={}),
        )
        assert resp.status_code == 422
        mock_service.predict.assert_not_awaited()

    def test_invalid_model_name_characters_returns_422(self, client, mock_service):
        """model_name with spaces or special chars fails _MODEL_NAME_RE."""
        for bad_name in ("my model!", "model name", "model@v1", "model/v1"):
            resp = client.post(
                "/inference/predict",
                json=_valid_payload(model_name=bad_name),
            )
            assert resp.status_code == 422, f"'{bad_name}' should be rejected"

    def test_invalid_version_string_returns_422(self, client, mock_service):
        """Arbitrary version strings that don't match semver or stage aliases."""
        for bad_ver in ("v1", "1.2", "release-1", "PROD"):
            resp = client.post(
                "/inference/predict",
                json=_valid_payload(version=bad_ver),
            )
            assert resp.status_code == 422, f"version '{bad_ver}' should be rejected"

    def test_extra_unknown_field_returns_422(self, client, mock_service):
        """PredictionRequest has extra='forbid' — unknown fields must be rejected."""
        payload = _valid_payload()
        payload["unknown_field"] = "should_fail"
        resp = client.post("/inference/predict", json=payload)
        assert resp.status_code == 422
        mock_service.predict.assert_not_awaited()

    def test_features_exceeding_500_keys_returns_422(self, client, mock_service):
        """prediction_schema caps features at 500 keys."""
        big_features = {f"feature_{i}": i for i in range(501)}
        resp = client.post(
            "/inference/predict",
            json=_valid_payload(features=big_features),
        )
        assert resp.status_code == 422
        mock_service.predict.assert_not_awaited()

    def test_invalid_pipeline_type_returns_422(self, client, mock_service):
        """pipeline_type must be one of PipelineType enum values."""
        resp = client.post(
            "/inference/predict",
            json=_valid_payload(pipeline_type="neural_net"),
        )
        assert resp.status_code == 422
        mock_service.predict.assert_not_awaited()

    def test_empty_model_name_returns_422(self, client, mock_service):
        """model_name has min_length=1 — empty string must fail."""
        resp = client.post(
            "/inference/predict",
            json=_valid_payload(model_name=""),
        )
        assert resp.status_code == 422
        mock_service.predict.assert_not_awaited()


class TestPredictErrorHandling:

    def test_value_error_returns_404(self, client, mock_service):
        """
        service.predict raises ValueError (model not found / invalid features)
        → route must return 404 with the error detail.
        """
        mock_service.predict = AsyncMock(
            side_effect=ValueError("Model 'missing_model' not found in registry.")
        )
        resp = client.post("/inference/predict", json=_valid_payload())
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_value_error_detail_matches_exception_message(self, client, mock_service):
        """The 404 detail string must be the ValueError message."""
        msg = "Invalid features: column 'income' is missing."
        mock_service.predict = AsyncMock(side_effect=ValueError(msg))
        resp = client.post("/inference/predict", json=_valid_payload())
        assert resp.json()["detail"] == msg

    def test_timeout_error_returns_504(self, client, mock_service):
        """
        asyncio.TimeoutError from service (pipeline exceeded 30s timeout)
        → route must return 504 Gateway Timeout.
        """
        mock_service.predict = AsyncMock(
            side_effect=asyncio.TimeoutError("Pipeline predict() exceeded 30s timeout.")
        )
        resp = client.post("/inference/predict", json=_valid_payload())
        assert resp.status_code == 504

    def test_timeout_error_detail_mentions_timeout(self, client, mock_service):
        """504 response body must mention timeout."""
        mock_service.predict = AsyncMock(side_effect=asyncio.TimeoutError())
        resp = client.post("/inference/predict", json=_valid_payload())
        assert "timed out" in resp.json()["detail"].lower()

    def test_unexpected_exception_returns_500(self, client, mock_service):
        """
        Any unexpected exception (S3 down, registry crash) → 500.
        """
        mock_service.predict = AsyncMock(
            side_effect=RuntimeError("S3 connection refused.")
        )
        resp = client.post("/inference/predict", json=_valid_payload())
        assert resp.status_code == 500

    def test_500_detail_is_generic(self, client, mock_service):
        """
        Internal error detail must NOT expose the raw exception message
        (security: don't leak stack traces to clients).
        """
        mock_service.predict = AsyncMock(
            side_effect=RuntimeError("db password: hunter2")
        )
        resp = client.post("/inference/predict", json=_valid_payload())
        body = resp.json()
        assert "hunter2" not in body["detail"]
        assert "internal error" in body["detail"].lower()

    def test_service_called_exactly_once_per_request(self, client, mock_service):
        """Exactly one service.predict() call per HTTP request — no double calls."""
        client.post("/inference/predict", json=_valid_payload())
        assert mock_service.predict.await_count == 1


class TestTraceContext:

    def test_trace_id_forwarded_to_service(self, client, mock_service):
        """
        trace_id from PredictionRequest.trace must be forwarded into the
        ServiceRequest that gets passed to service.predict().
        """
        custom_trace = {"trace_id": "abc123fixed", "span_id": "span999"}
        payload = _valid_payload()
        payload["trace"] = custom_trace

        client.post("/inference/predict", json=payload)

        # Inspect what ServiceRequest was passed to service.predict
        call_args = mock_service.predict.call_args
        service_req = call_args[0][0]   # first positional arg
        assert service_req.trace_id == "abc123fixed"

    def test_auto_generated_trace_id_is_unique(self, client, mock_service):
        """Two requests without explicit trace → each gets a different trace_id."""
        client.post("/inference/predict", json=_valid_payload())
        client.post("/inference/predict", json=_valid_payload())

        calls = mock_service.predict.call_args_list
        trace_id_1 = calls[0][0][0].trace_id
        trace_id_2 = calls[1][0][0].trace_id
        assert trace_id_1 != trace_id_2

    def test_response_trace_matches_request_trace(self, client, mock_service):
        """trace in response must echo back the trace sent in the request."""
        custom_trace = {"trace_id": "mytrace001", "span_id": "myspan001"}
        payload = _valid_payload()
        payload["trace"] = custom_trace

        body = client.post("/inference/predict", json=payload).json()
        assert body["trace"]["trace_id"] == "mytrace001"



class TestPipelineTypeBridge:
    """
    The route converts PipelineType enum → string value before passing
    to the service dataclass (which expects "tabular" / "temporal").
    """

    def test_tabular_enum_bridged_as_string(self, client, mock_service):
        client.post("/inference/predict", json=_valid_payload(pipeline_type="tabular"))
        service_req = mock_service.predict.call_args[0][0]
        assert service_req.pipeline_type == "tabular"

    def test_temporal_enum_bridged_as_string(self, client, mock_service):
        client.post("/inference/predict", json=_valid_payload(pipeline_type="temporal"))
        service_req = mock_service.predict.call_args[0][0]
        assert service_req.pipeline_type == "temporal"

    def test_features_passed_unchanged(self, client, mock_service):
        """features dict must reach the service exactly as sent."""
        features = {"age": 25, "income": 75000, "tenure": 3}
        client.post("/inference/predict", json=_valid_payload(features=features))
        service_req = mock_service.predict.call_args[0][0]
        assert service_req.features == features

    def test_model_name_passed_unchanged(self, client, mock_service):
        client.post(
            "/inference/predict",
            json=_valid_payload(model_name="churn-predictor-v2"),
        )
        service_req = mock_service.predict.call_args[0][0]
        assert service_req.model_name == "churn-predictor-v2"



class TestHealthEndpoint:

    def test_health_returns_200_when_all_healthy(self, client, mock_service):
        mock_service.health.return_value = {"redis": True, "postgres": True}
        resp = client.get("/inference/health")
        assert resp.status_code == 200

    def test_health_body_contains_redis_and_postgres(self, client, mock_service):
        mock_service.health.return_value = {"redis": True, "postgres": True}
        body = client.get("/inference/health").json()
        assert "redis"    in body
        assert "postgres" in body

    def test_health_reflects_redis_down(self, client, mock_service):
        """If Redis is down the response must reflect redis: False."""
        mock_service.health.return_value = {"redis": False, "postgres": True}
        body = client.get("/inference/health").json()
        assert body["redis"]    is False
        assert body["postgres"] is True

    def test_health_reflects_postgres_down(self, client, mock_service):
        mock_service.health.return_value = {"redis": True, "postgres": False}
        body = client.get("/inference/health").json()
        assert body["redis"]    is True
        assert body["postgres"] is False

    def test_health_calls_service_health_once(self, client, mock_service):
        """Exactly one call to service.health() per /health request."""
        client.get("/inference/health")
        mock_service.health.assert_called_once()