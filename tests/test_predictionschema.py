"""
tests/test_prediction_schema.py
================================
Full test suite for prediction_schema.py.

Coverage map
------------
1.  TraceContext                   — defaults, explicit values, immutability
2.  PredictionError                — all fields, retryable flag, UTC timestamp
3.  compute_cache_key              — stability, order-independence, sensitivity,
                                     None/latest equivalence, cross-process safety
4.  PredictionRequest — happy path — all fields, defaults, UUID auto-gen
5.  PredictionRequest — model_name — valid names, invalid names (spaces, dots,
                                     slashes, @, empty string)
6.  PredictionRequest — version    — all valid semver + stage aliases,
                                     invalid formats
7.  PredictionRequest — features   — empty dict, too many keys, oversized key,
                                     valid large dict
8.  PredictionRequest — cache_key  — auto-computed, stable, matches helper
9.  PredictionRequest — extra="forbid"
10. PredictionRequest — requested_at is UTC-aware
11. PredictionResponse             — happy path, cached flag, structured error,
                                     confidence bounds, to_dict(), frozen
12. BatchPredictionRequest         — happy path, priority bounds, cache_keys
                                     property, empty input, oversized batch,
                                     empty item, extra="forbid"
13. BatchItemResult                — success item, failed item, cached item
14. BatchPredictionResponse.from_results — all-success, all-failed, partial,
                                     cached count, counter consistency check,
                                     trace propagation
15. BatchPredictionResponse        — inconsistent counters raise ValueError
16. PipelineType enum              — all values round-trip
17. PredictionStatus enum          — all values
18. ErrorCode enum                 — all values
19. Integration: request → cache_key → batch cache_keys consistency
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from uuid import UUID

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCHEMA_DIR = os.path.normpath(os.path.join(_HERE, "..", "outputs"))
if _SCHEMA_DIR not in sys.path:
    sys.path.insert(0, _SCHEMA_DIR)

from app.schemas.prediction_schema import (   # noqa: E402
    BatchItemResult,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorCode,
    PipelineType,
    PredictionError,
    PredictionRequest,
    PredictionResponse,
    PredictionStatus,
    TraceContext,
    _MAX_BATCH_SIZE,
    _MAX_FEATURE_KEY_LEN,
    _MAX_FEATURES,
    _MAX_MODEL_NAME_LEN,
    compute_cache_key,
)
from pydantic import ValidationError



@pytest.fixture
def basic_features():
    return {"age": 35, "income": 75_000.0, "score": 0.82}


@pytest.fixture
def prediction_request(basic_features):
    return PredictionRequest(
        model_name="churn_predictor",
        features=basic_features,
    )


@pytest.fixture
def batch_request(basic_features):
    return BatchPredictionRequest(
        model_name="churn_predictor",
        inputs=[basic_features, {"age": 40, "income": 90_000.0, "score": 0.91}],
    )


@pytest.fixture
def success_item():
    return BatchItemResult(index=0, status=PredictionStatus.SUCCESS, prediction=0.85)


@pytest.fixture
def failed_item():
    return BatchItemResult(
        index=1,
        status=PredictionStatus.FAILED,
        error=PredictionError(
            error_code=ErrorCode.INTERNAL_ERROR,
            message="model exploded",
        ),
    )



class TestTraceContext:

    def test_defaults_auto_generated(self):
        t = TraceContext()
        assert isinstance(t.trace_id, str) and len(t.trace_id) == 32   # uuid4().hex
        assert isinstance(t.span_id,  str) and len(t.span_id)  == 16
        assert t.parent_span is None
        assert t.source      is None

    def test_explicit_values(self):
        t = TraceContext(
            trace_id="abc123",
            span_id="span0001",
            parent_span="parent99",
            source="api_gateway",
        )
        assert t.trace_id    == "abc123"
        assert t.span_id     == "span0001"
        assert t.parent_span == "parent99"
        assert t.source      == "api_gateway"

    def test_two_instances_have_unique_ids(self):
        t1, t2 = TraceContext(), TraceContext()
        assert t1.trace_id != t2.trace_id
        assert t1.span_id  != t2.span_id

    def test_frozen(self):
        """TraceContext is immutable after construction."""
        t = TraceContext()
        with pytest.raises(Exception):   # ValidationError or TypeError
            t.trace_id = "tampered"



class TestPredictionError:

    def test_required_fields(self):
        e = PredictionError(
            error_code=ErrorCode.MODEL_NOT_FOUND,
            message="Model churn_v2 does not exist",
        )
        assert e.error_code  == ErrorCode.MODEL_NOT_FOUND
        assert e.message     == "Model churn_v2 does not exist"
        assert e.retryable   is False
        assert e.detail      is None

    def test_retryable_true(self):
        e = PredictionError(
            error_code=ErrorCode.UPSTREAM_FAILURE,
            message="Redis timeout",
            retryable=True,
        )
        assert e.retryable is True

    def test_detail_dict(self):
        e = PredictionError(
            error_code=ErrorCode.FEATURE_MISMATCH,
            message="Missing features",
            detail={"missing": ["age", "income"]},
        )
        assert e.detail["missing"] == ["age", "income"]

    def test_occurred_at_is_utc_aware(self):
        e = PredictionError(error_code=ErrorCode.TIMEOUT, message="timed out")
        assert e.occurred_at.tzinfo is not None

    def test_frozen(self):
        e = PredictionError(error_code=ErrorCode.INTERNAL_ERROR, message="x")
        with pytest.raises(Exception):
            e.message = "tampered"



class TestComputeCacheKey:

    def test_returns_64_char_hex_string(self):
        k = compute_cache_key("model", "1.0.0", {"a": 1})
        assert isinstance(k, str)
        assert len(k) == 64
        assert all(c in "0123456789abcdef" for c in k)

    def test_stable_across_calls(self):
        f = {"x": 1, "y": 2}
        assert compute_cache_key("m", "1.0.0", f) == compute_cache_key("m", "1.0.0", f)

    def test_order_independent(self):
        """Dict insertion order must not affect the key."""
        k1 = compute_cache_key("m", "1.0.0", {"a": 1, "b": 2})
        k2 = compute_cache_key("m", "1.0.0", {"b": 2, "a": 1})
        assert k1 == k2

    def test_none_version_equals_latest(self):
        f = {"a": 1}
        assert compute_cache_key("m", None, f) == compute_cache_key("m", "latest", f)

    def test_different_models_differ(self):
        f = {"a": 1}
        assert compute_cache_key("model_a", "1.0.0", f) != compute_cache_key("model_b", "1.0.0", f)

    def test_different_versions_differ(self):
        f = {"a": 1}
        assert compute_cache_key("m", "1.0.0", f) != compute_cache_key("m", "2.0.0", f)

    def test_different_features_differ(self):
        assert compute_cache_key("m", "1.0.0", {"a": 1}) != compute_cache_key("m", "1.0.0", {"a": 2})

    def test_different_feature_keys_differ(self):
        assert compute_cache_key("m", "1.0.0", {"a": 1}) != compute_cache_key("m", "1.0.0", {"b": 1})

    def test_matches_manual_sha256(self):
        """Key must equal what you'd get computing SHA-256 by hand."""
        model, version, features = "churn", "1.0.0", {"age": 35}
        payload = json.dumps(
            {"model": model, "version": version, "features": features},
            sort_keys=True, default=str,
        )
        expected = hashlib.sha256(payload.encode()).hexdigest()
        assert compute_cache_key(model, version, features) == expected

    def test_non_serialisable_value_handled(self):
        """Non-JSON types must not raise — they are coerced via default=str."""
        from datetime import date
        k = compute_cache_key("m", "1.0.0", {"dt": date(2024, 1, 1)})
        assert len(k) == 64



class TestPredictionRequestHappyPath:

    def test_minimal_required_fields(self, basic_features):
        req = PredictionRequest(model_name="my_model", features=basic_features)
        assert req.model_name == "my_model"
        assert req.features   == basic_features

    def test_request_id_auto_generated_uuid(self, prediction_request):
        assert isinstance(prediction_request.request_id, UUID)

    def test_two_requests_have_different_ids(self, basic_features):
        r1 = PredictionRequest(model_name="m", features=basic_features)
        r2 = PredictionRequest(model_name="m", features=basic_features)
        assert r1.request_id != r2.request_id

    def test_version_defaults_to_none(self, prediction_request):
        assert prediction_request.version is None

    def test_pipeline_type_defaults_to_tabular(self, prediction_request):
        assert prediction_request.pipeline_type == PipelineType.TABULAR

    def test_explicit_pipeline_type(self, basic_features):
        req = PredictionRequest(
            model_name="m",
            features=basic_features,
            pipeline_type=PipelineType.TEMPORAL,
        )
        assert req.pipeline_type == PipelineType.TEMPORAL

    def test_trace_context_auto_created(self, prediction_request):
        assert isinstance(prediction_request.trace, TraceContext)
        assert isinstance(prediction_request.trace.trace_id, str)

    def test_explicit_trace_context(self, basic_features):
        trace = TraceContext(trace_id="fixed-trace", source="api_gateway")
        req   = PredictionRequest(model_name="m", features=basic_features, trace=trace)
        assert req.trace.trace_id == "fixed-trace"
        assert req.trace.source   == "api_gateway"

    def test_requested_at_is_utc_aware(self, prediction_request):
        assert prediction_request.requested_at.tzinfo is not None
        assert prediction_request.requested_at.tzinfo == timezone.utc

    def test_all_valid_pipeline_types(self, basic_features):
        for pt in PipelineType:
            req = PredictionRequest(model_name="m", features=basic_features, pipeline_type=pt)
            assert req.pipeline_type == pt



class TestPredictionRequestModelName:

    @pytest.mark.parametrize("name", [
        "my_model",
        "churn-predictor-v2",
        "ModelABC",
        "a1",
        "a-b_c",
        "sales_forecast_model",
        "x" * _MAX_MODEL_NAME_LEN,   # exactly at the max
    ])
    def test_valid_model_names(self, name, basic_features):
        req = PredictionRequest(model_name=name, features=basic_features)
        assert req.model_name == name

    @pytest.mark.parametrize("name, reason", [
        ("model name",  "space"),
        ("model.name",  "dot"),
        ("model/name",  "slash"),
        ("model@v1",    "at-sign"),
        ("model!",      "exclamation"),
        ("",            "empty string — min_length=1"),
        ("x" * (_MAX_MODEL_NAME_LEN + 1), "exceeds max length"),
    ])
    def test_invalid_model_names_raise(self, name, reason, basic_features):
        with pytest.raises(ValidationError, match=r"(?i)(model_name|string_too_long|too short)"):
            PredictionRequest(model_name=name, features=basic_features)



class TestPredictionRequestVersion:

    @pytest.mark.parametrize("version", [
        "1.0.0",
        "2.3.11",
        "0.0.1",
        "1.0.0-rc1",
        "1.0.0-beta",
        "latest",
        "staging",
        "production",
        None,             # version is optional
    ])
    def test_valid_versions(self, version, basic_features):
        req = PredictionRequest(
            model_name="m", features=basic_features, version=version
        )
        assert req.version == version

    @pytest.mark.parametrize("version", [
        "1.0",
        "v1.0.0",
        "1.0.0.0",
        "main",
        "dev",
        "abc",
        "1.0.0_rc1",
        "LATEST",
        "Staging",
    ])
    def test_invalid_versions_raise(self, version, basic_features):
        with pytest.raises(ValidationError):
            PredictionRequest(model_name="m", features=basic_features, version=version)



class TestPredictionRequestFeatures:

    def test_empty_features_raises(self):
        with pytest.raises(ValidationError, match="must not be empty"):
            PredictionRequest(model_name="m", features={})

    def test_too_many_features_raises(self):
        big = {str(i): i for i in range(_MAX_FEATURES + 1)}
        with pytest.raises(ValidationError, match=str(_MAX_FEATURES)):
            PredictionRequest(model_name="m", features=big)

    def test_exactly_max_features_allowed(self):
        exactly = {str(i): i for i in range(_MAX_FEATURES)}
        req = PredictionRequest(model_name="m", features=exactly)
        assert len(req.features) == _MAX_FEATURES

    def test_oversized_key_raises(self):
        long_key = "x" * (_MAX_FEATURE_KEY_LEN + 1)
        with pytest.raises(ValidationError, match=str(_MAX_FEATURE_KEY_LEN)):
            PredictionRequest(model_name="m", features={long_key: 1})

    def test_exactly_max_key_length_allowed(self):
        max_key = "x" * _MAX_FEATURE_KEY_LEN
        req = PredictionRequest(model_name="m", features={max_key: 1})
        assert max_key in req.features

    def test_various_value_types_accepted(self):
        req = PredictionRequest(
            model_name="m",
            features={"int": 1, "float": 1.5, "str": "a", "bool": True, "list": [1, 2], "none": None},
        )
        assert req.features["list"] == [1, 2]



class TestPredictionRequestCacheKey:

    def test_cache_key_auto_set(self, prediction_request):
        assert prediction_request.cache_key is not None
        assert len(prediction_request.cache_key) == 64

    def test_cache_key_matches_helper(self, basic_features):
        req = PredictionRequest(model_name="churn_predictor", features=basic_features)
        expected = compute_cache_key("churn_predictor", None, basic_features)
        assert req.cache_key == expected

    def test_cache_key_includes_version(self, basic_features):
        r1 = PredictionRequest(model_name="m", features=basic_features, version="1.0.0")
        r2 = PredictionRequest(model_name="m", features=basic_features, version="2.0.0")
        assert r1.cache_key != r2.cache_key

    def test_same_inputs_same_cache_key(self, basic_features):
        r1 = PredictionRequest(model_name="m", features=basic_features)
        r2 = PredictionRequest(model_name="m", features=basic_features)
        assert r1.cache_key == r2.cache_key

    def test_different_features_different_cache_key(self):
        r1 = PredictionRequest(model_name="m", features={"a": 1})
        r2 = PredictionRequest(model_name="m", features={"a": 2})
        assert r1.cache_key != r2.cache_key

    def test_none_and_latest_version_produce_same_key(self, basic_features):
        r1 = PredictionRequest(model_name="m", features=basic_features, version=None)
        r2 = PredictionRequest(model_name="m", features=basic_features, version="latest")
        assert r1.cache_key == r2.cache_key


class TestPredictionRequestExtraForbid:

    def test_extra_field_raises(self, basic_features):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            PredictionRequest(
                model_name="m",
                features=basic_features,
                unknown_field="bad",
            )



class TestPredictionRequestTimestamp:

    def test_requested_at_has_utc_timezone(self, prediction_request):
        ts = prediction_request.requested_at
        assert ts.tzinfo is not None
        assert ts.utcoffset().total_seconds() == 0

    def test_requested_at_is_recent(self, prediction_request):
        now  = datetime.now(timezone.utc)
        diff = abs((now - prediction_request.requested_at).total_seconds())
        assert diff < 5.0, "requested_at should be within 5 seconds of now"



class TestPredictionResponse:

    def _make(self, **kw):
        defaults = dict(
            request_id    = __import__("uuid").uuid4(),
            trace         = TraceContext(),
            model_name    = "churn_predictor",
            model_version = "1.0.0",
            status        = PredictionStatus.SUCCESS,
            prediction    = 0.87,
            latency_ms    = 12.5,
        )
        defaults.update(kw)
        return PredictionResponse(**defaults)

    def test_happy_path(self):
        resp = self._make()
        assert resp.model_name    == "churn_predictor"
        assert resp.status        == PredictionStatus.SUCCESS
        assert resp.prediction    == 0.87
        assert resp.cached        is False
        assert resp.error         is None

    def test_cached_flag(self):
        resp = self._make(cached=True, status=PredictionStatus.CACHED)
        assert resp.cached is True

    def test_confidence_within_bounds(self):
        resp = self._make(confidence=0.95)
        assert resp.confidence == 0.95

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValidationError):
            self._make(confidence=-0.01)

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValidationError):
            self._make(confidence=1.001)

    def test_confidence_boundary_values(self):
        resp_lo = self._make(confidence=0.0)
        resp_hi = self._make(confidence=1.0)
        assert resp_lo.confidence == 0.0
        assert resp_hi.confidence == 1.0

    def test_structured_error_attached(self):
        err  = PredictionError(error_code=ErrorCode.TIMEOUT, message="slow")
        resp = self._make(
            status     = PredictionStatus.FAILED,
            prediction = None,
            error      = err,
        )
        assert resp.error.error_code == ErrorCode.TIMEOUT
        assert resp.error.message    == "slow"

    def test_to_dict_returns_json_safe_dict(self):
        resp = self._make()
        d    = resp.to_dict()
        assert isinstance(d, dict)
        assert d["model_name"] == "churn_predictor"
        # UUID and datetime must be serialised (not raw objects)
        assert not isinstance(d["request_id"], __import__("uuid").UUID)

    def test_responded_at_is_utc_aware(self):
        resp = self._make()
        assert resp.responded_at.tzinfo is not None

    def test_frozen_response_cannot_be_mutated(self):
        resp = self._make()
        with pytest.raises(Exception):
            resp.prediction = 0.99

    def test_trace_propagated(self):
        trace = TraceContext(trace_id="t123", source="gateway")
        resp  = self._make(trace=trace)
        assert resp.trace.trace_id == "t123"
        assert resp.trace.source   == "gateway"

    def test_negative_latency_raises(self):
        with pytest.raises(ValidationError):
            self._make(latency_ms=-1.0)



class TestBatchPredictionRequest:

    def test_happy_path(self, batch_request):
        assert batch_request.model_name   == "churn_predictor"
        assert len(batch_request.inputs)  == 2
        assert batch_request.priority     == 5

    def test_request_id_auto_generated(self, batch_request):
        assert isinstance(batch_request.request_id, UUID)

    def test_pipeline_type_defaults_tabular(self, batch_request):
        assert batch_request.pipeline_type == PipelineType.TABULAR

    def test_priority_minimum(self, basic_features):
        req = BatchPredictionRequest(
            model_name="m", inputs=[basic_features], priority=1
        )
        assert req.priority == 1

    def test_priority_maximum(self, basic_features):
        req = BatchPredictionRequest(
            model_name="m", inputs=[basic_features], priority=10
        )
        assert req.priority == 10

    def test_priority_below_min_raises(self, basic_features):
        with pytest.raises(ValidationError):
            BatchPredictionRequest(model_name="m", inputs=[basic_features], priority=0)

    def test_priority_above_max_raises(self, basic_features):
        with pytest.raises(ValidationError):
            BatchPredictionRequest(model_name="m", inputs=[basic_features], priority=11)

    def test_empty_inputs_list_raises(self):
        with pytest.raises(ValidationError):
            BatchPredictionRequest(model_name="m", inputs=[])

    def test_batch_size_at_max_allowed(self, basic_features):
        big = [basic_features] * _MAX_BATCH_SIZE
        req = BatchPredictionRequest(model_name="m", inputs=big)
        assert len(req.inputs) == _MAX_BATCH_SIZE

    def test_batch_size_exceeds_max_raises(self, basic_features):
        too_big = [basic_features] * (_MAX_BATCH_SIZE + 1)
        with pytest.raises(ValidationError, match=str(_MAX_BATCH_SIZE)):
            BatchPredictionRequest(model_name="m", inputs=too_big)

    def test_empty_item_in_inputs_raises(self, basic_features):
        with pytest.raises(ValidationError, match=r"inputs\[1\]"):
            BatchPredictionRequest(model_name="m", inputs=[basic_features, {}])

    def test_extra_field_forbidden(self, basic_features):
        with pytest.raises(ValidationError):
            BatchPredictionRequest(model_name="m", inputs=[basic_features], rogue="bad")

    def test_valid_version(self, basic_features):
        req = BatchPredictionRequest(
            model_name="m", inputs=[basic_features], version="production"
        )
        assert req.version == "production"

    def test_invalid_version_raises(self, basic_features):
        with pytest.raises(ValidationError):
            BatchPredictionRequest(model_name="m", inputs=[basic_features], version="main")

    def test_trace_auto_created(self, batch_request):
        assert isinstance(batch_request.trace, TraceContext)

    def test_requested_at_utc_aware(self, batch_request):
        assert batch_request.requested_at.tzinfo is not None



class TestBatchCacheKeys:

    def test_cache_keys_length_matches_inputs(self, batch_request):
        keys = batch_request.cache_keys
        assert len(keys) == len(batch_request.inputs)

    def test_each_key_is_64_char_hex(self, batch_request):
        for k in batch_request.cache_keys:
            assert len(k) == 64
            assert all(c in "0123456789abcdef" for c in k)

    def test_cache_keys_match_helper(self, batch_request):
        expected = [
            compute_cache_key(batch_request.model_name, batch_request.version, item)
            for item in batch_request.inputs
        ]
        assert batch_request.cache_keys == expected

    def test_different_items_different_keys(self, basic_features):
        req = BatchPredictionRequest(
            model_name="m",
            inputs=[{"a": 1}, {"a": 2}],
        )
        assert req.cache_keys[0] != req.cache_keys[1]

    def test_identical_items_same_key(self):
        req = BatchPredictionRequest(
            model_name="m",
            inputs=[{"a": 1}, {"a": 1}],
        )
        assert req.cache_keys[0] == req.cache_keys[1]

    def test_cache_keys_stable_across_calls(self, batch_request):
        assert batch_request.cache_keys == batch_request.cache_keys



class TestBatchItemResult:

    def test_success_item(self, success_item):
        assert success_item.index      == 0
        assert success_item.status     == PredictionStatus.SUCCESS
        assert success_item.prediction == 0.85
        assert success_item.cached     is False
        assert success_item.error      is None

    def test_failed_item_with_error(self, failed_item):
        assert failed_item.status            == PredictionStatus.FAILED
        assert failed_item.error.error_code  == ErrorCode.INTERNAL_ERROR
        assert failed_item.prediction        is None

    def test_cached_item(self):
        item = BatchItemResult(
            index=2, status=PredictionStatus.SUCCESS, prediction=0.5, cached=True
        )
        assert item.cached is True

    def test_confidence_bounds(self):
        item = BatchItemResult(index=0, status=PredictionStatus.SUCCESS, confidence=0.75)
        assert item.confidence == 0.75

    def test_confidence_out_of_bounds_raises(self):
        with pytest.raises(ValidationError):
            BatchItemResult(index=0, status=PredictionStatus.SUCCESS, confidence=1.5)

    def test_frozen(self, success_item):
        with pytest.raises(Exception):
            success_item.prediction = 0.0



class TestBatchPredictionResponseFromResults:

    def _make_batch_req(self):
        return BatchPredictionRequest(
            model_name="churn_predictor",
            inputs=[{"a": 1}, {"a": 2}, {"a": 3}],
        )

    def test_all_success_status(self):
        req     = self._make_batch_req()
        results = [
            BatchItemResult(index=i, status=PredictionStatus.SUCCESS, prediction=0.9)
            for i in range(3)
        ]
        resp = BatchPredictionResponse.from_results(
            request=req, results=results, model_version="1.0.0", latency_ms=50.0
        )
        assert resp.status    == PredictionStatus.SUCCESS
        assert resp.succeeded == 3
        assert resp.failed    == 0
        assert resp.total     == 3

    def test_all_failed_status(self):
        req     = self._make_batch_req()
        results = [
            BatchItemResult(
                index=i, status=PredictionStatus.FAILED,
                error=PredictionError(error_code=ErrorCode.INTERNAL_ERROR, message="x")
            )
            for i in range(3)
        ]
        resp = BatchPredictionResponse.from_results(
            request=req, results=results, model_version="1.0.0", latency_ms=10.0
        )
        assert resp.status    == PredictionStatus.FAILED
        assert resp.succeeded == 0
        assert resp.failed    == 3

    def test_partial_status(self):
        req     = self._make_batch_req()
        results = [
            BatchItemResult(index=0, status=PredictionStatus.SUCCESS, prediction=0.9),
            BatchItemResult(
                index=1, status=PredictionStatus.FAILED,
                error=PredictionError(error_code=ErrorCode.FEATURE_MISMATCH, message="x")
            ),
            BatchItemResult(index=2, status=PredictionStatus.SUCCESS, prediction=0.7),
        ]
        resp = BatchPredictionResponse.from_results(
            request=req, results=results, model_version="1.0.0", latency_ms=30.0
        )
        assert resp.status    == PredictionStatus.PARTIAL
        assert resp.succeeded == 2
        assert resp.failed    == 1
        assert resp.total     == 3

    def test_cached_count(self):
        req = BatchPredictionRequest(model_name="m", inputs=[{"a": i} for i in range(4)])
        results = [
            BatchItemResult(index=0, status=PredictionStatus.SUCCESS, prediction=1, cached=True),
            BatchItemResult(index=1, status=PredictionStatus.SUCCESS, prediction=2, cached=True),
            BatchItemResult(index=2, status=PredictionStatus.SUCCESS, prediction=3, cached=False),
            BatchItemResult(index=3, status=PredictionStatus.SUCCESS, prediction=4, cached=False),
        ]
        resp = BatchPredictionResponse.from_results(
            request=req, results=results, model_version="1.0.0", latency_ms=5.0
        )
        assert resp.cached    == 2
        assert resp.succeeded == 4

    def test_trace_propagated_from_request(self):
        trace = TraceContext(trace_id="batch-trace-001", source="batch_worker")
        req   = BatchPredictionRequest(
            model_name="m", inputs=[{"a": 1}], trace=trace
        )
        results = [BatchItemResult(index=0, status=PredictionStatus.SUCCESS, prediction=1)]
        resp    = BatchPredictionResponse.from_results(
            request=req, results=results, model_version="1.0.0", latency_ms=1.0
        )
        assert resp.trace.trace_id == "batch-trace-001"
        assert resp.trace.source   == "batch_worker"

    def test_request_id_propagated(self):
        req     = self._make_batch_req()
        results = [BatchItemResult(index=0, status=PredictionStatus.SUCCESS, prediction=1)]
        resp    = BatchPredictionResponse.from_results(
            request=req, results=results, model_version="1.0.0", latency_ms=1.0
        )
        assert resp.request_id == req.request_id

    def test_model_version_set(self):
        req     = self._make_batch_req()
        results = [BatchItemResult(index=0, status=PredictionStatus.SUCCESS, prediction=1)]
        resp    = BatchPredictionResponse.from_results(
            request=req, results=results, model_version="2.3.0", latency_ms=1.0
        )
        assert resp.model_version == "2.3.0"

    def test_latency_ms_set(self):
        req     = self._make_batch_req()
        results = [BatchItemResult(index=0, status=PredictionStatus.SUCCESS, prediction=1)]
        resp    = BatchPredictionResponse.from_results(
            request=req, results=results, model_version="1.0.0", latency_ms=142.7
        )
        assert resp.latency_ms == 142.7

    def test_to_dict_returns_json_safe_dict(self):
        req     = self._make_batch_req()
        results = [BatchItemResult(index=0, status=PredictionStatus.SUCCESS, prediction=1)]
        resp    = BatchPredictionResponse.from_results(
            request=req, results=results, model_version="1.0.0", latency_ms=1.0
        )
        d = resp.to_dict()
        assert isinstance(d, dict)
        assert d["succeeded"] == 1
        assert not isinstance(d["request_id"], __import__("uuid").UUID)



class TestBatchPredictionResponseCounterValidation:

    def test_inconsistent_counters_raise(self):
        """succeeded + failed must equal total — validator catches mismatches."""
        import uuid
        with pytest.raises(ValidationError, match="must equal total"):
            BatchPredictionResponse(
                request_id    = uuid.uuid4(),
                trace         = TraceContext(),
                model_name    = "m",
                model_version = "1.0.0",
                status        = PredictionStatus.SUCCESS,
                results       = [],
                total         = 5,     #  wrong: 2 + 2 ≠ 5
                succeeded     = 2,
                failed        = 2,
                cached        = 0,
                latency_ms    = 1.0,
            )

    def test_consistent_counters_accepted(self):
        import uuid
        resp = BatchPredictionResponse(
            request_id    = uuid.uuid4(),
            trace         = TraceContext(),
            model_name    = "m",
            model_version = "1.0.0",
            status        = PredictionStatus.SUCCESS,
            results       = [],
            total         = 4,
            succeeded     = 3,
            failed        = 1,
            cached        = 0,
            latency_ms    = 1.0,
        )
        assert resp.total == 4



class TestEnums:

    def test_pipeline_type_values(self):
        assert PipelineType.TEMPORAL.value    == "temporal"
        assert PipelineType.TABULAR.value     == "tabular"
        assert PipelineType.FORECASTING.value == "forecasting"

    def test_prediction_status_values(self):
        assert PredictionStatus.SUCCESS.value == "success"
        assert PredictionStatus.FAILED.value  == "failed"
        assert PredictionStatus.TIMEOUT.value == "timeout"
        assert PredictionStatus.CACHED.value  == "cached"
        assert PredictionStatus.PARTIAL.value == "partial"

    def test_error_code_values(self):
        codes = {e.value for e in ErrorCode}
        assert "VALIDATION_ERROR"    in codes
        assert "MODEL_NOT_FOUND"     in codes
        assert "FEATURE_MISMATCH"    in codes
        assert "TIMEOUT"             in codes
        assert "DRIFT_DETECTED"      in codes
        assert "UPSTREAM_FAILURE"    in codes
        assert "INTERNAL_ERROR"      in codes
        assert "BATCH_SIZE_EXCEEDED" in codes
        assert "RATE_LIMITED"        in codes

    def test_pipeline_type_roundtrip(self):
        for pt in PipelineType:
            assert PipelineType(pt.value) == pt

    def test_prediction_status_roundtrip(self):
        for ps in PredictionStatus:
            assert PredictionStatus(ps.value) == ps



class TestIntegration:

    def test_single_and_batch_cache_keys_match(self, basic_features):
        """
        A PredictionRequest and a BatchPredictionRequest for the same
        model + version + features must produce the same cache key.
        This guarantees that a cached single prediction is served for
        the matching batch item and vice versa.
        """
        single = PredictionRequest(
            model_name="churn_predictor",
            features=basic_features,
            version="1.0.0",
        )
        batch = BatchPredictionRequest(
            model_name="churn_predictor",
            inputs=[basic_features],
            version="1.0.0",
        )
        assert single.cache_key == batch.cache_keys[0]

    def test_full_batch_lifecycle(self, basic_features):
        """
        Simulate the full batch flow:
        request → dispatch → per-item results → BatchPredictionResponse.
        """
        req = BatchPredictionRequest(
            model_name="churn_predictor",
            inputs=[basic_features, {"age": 50, "income": 120_000.0, "score": 0.95}],
            version="production",
        )

        # Simulate two items: first from cache, second from model
        results = [
            BatchItemResult(
                index=0, status=PredictionStatus.SUCCESS,
                prediction=0.88, confidence=0.92, cached=True,
            ),
            BatchItemResult(
                index=1, status=PredictionStatus.SUCCESS,
                prediction=0.72, confidence=0.85, cached=False,
            ),
        ]

        resp = BatchPredictionResponse.from_results(
            request=req, results=results, model_version="3.1.0", latency_ms=18.4
        )

        assert resp.status      == PredictionStatus.SUCCESS
        assert resp.total       == 2
        assert resp.succeeded   == 2
        assert resp.failed      == 0
        assert resp.cached      == 1
        assert resp.model_name  == "churn_predictor"
        assert resp.latency_ms  == 18.4

        # Verify to_dict is fully serialisable
        d = resp.to_dict()
        json.dumps(d)   # raises if not JSON-safe