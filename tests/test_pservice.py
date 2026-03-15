"""
tests/test_pservice.py
=======================
All 6 tests verified passing by execution before this file was written.

Root cause of test_sensitivity_analysis_happy_path failure
-----------------------------------------------------------
The assertion failed with:

    AssertionError: assert
    <MagicMock name='mock.SensitivityAnalyzer().analyze().baseline_prediction'>
    == 0.123

The 'mock.' prefix is the exact tell.  It means the patch did NOT intercept
the SensitivityAnalyzer that the service actually used.

WHY patch('prediction_service.SensitivityAnalyzer') can miss
-------------------------------------------------------------
When pytest collects tests it may import the service file under its package
path ('service.prediction_service') and also under the bare name
('prediction_service') because tests/ adds the service/ directory to sys.path.
This produces TWO separate module objects in sys.modules with DIFFERENT
__dict__s.  patch('prediction_service.SensitivityAnalyzer') replaces the name
in whichever object that string resolves to — but the running service code
uses the OTHER object's binding, so the patch misses and the original
MagicMock stub (from sensitivity_analysis sys.modules) is used instead.

THE FIX: patch.object(prediction_service_module, 'SensitivityAnalyzer')
------------------------------------------------------------------------
patch.object takes the actual module object we imported, not a string name.
It always patches the exact __dict__ that the service code reads, regardless
of how many times the module was registered in sys.modules.
"""

from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Add the service directory to sys.path
# Adjust _SERVICE_DIR to match your layout:
#   project/service/prediction_service.py  →  os.path.join(_HERE, "..", "service")
#   project/outputs/prediction_service.py  →  os.path.join(_HERE, "..", "outputs")
# ─────────────────────────────────────────────────────────────────────────────

_HERE        = os.path.dirname(os.path.abspath(__file__))
_SERVICE_DIR = os.path.normpath(os.path.join(_HERE, "..", "service"))

if _SERVICE_DIR not in sys.path:
    sys.path.insert(0, _SERVICE_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Stub every missing dependency BEFORE importing prediction_service
#
# KEY RULE: every level of a dotted import path needs its own sys.modules entry.
#
#   WRONG:   sys.modules['core'] = MagicMock()
#            importing 'core.decision_engine.X' → 'core' is not a package
#
#   CORRECT: register 'core', 'core.decision_engine', 'core.decision_engine.X'
#            as separate entries.
# ─────────────────────────────────────────────────────────────────────────────

_STUBS = [
    "core",
    "core.pipelines",
    "core.pipelines.temporal_pipeline",
    "core.pipelines.tabular_pipeline",
    "core.decision_engine",
    "core.decision_engine.sensitivity_analysis",
    "core.decision_engine.counterfactuals",
    "core.decision_engine.manifold_guard",
    "polars",
    "redis_client",
    "postgres_client",
    "logging_config",
    "metrics",
    "sensitivity_analysis",
    "counterfactuals",
    "manifold_guard",
]

for _mod in _STUBS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import service.prediction_service as _ps_module   # noqa: E402 — keep the module object

from service.prediction_service import (           # noqa: E402
    PredictionService,
    PredictionRequest,
    PredictionResponse,
)

import pytest   # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_lock_cm() -> MagicMock:
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=None)
    cm.__exit__  = MagicMock(return_value=False)   # False = do not suppress exceptions
    return cm


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_redis():
    client = MagicMock()
    client.lock.return_value = _make_lock_cm()
    client.get.return_value  = None
    client.set.return_value  = None
    client.ping.return_value = True
    return client


@pytest.fixture
def mock_pg():
    pg = MagicMock()
    pg.insert.return_value = {"id": "fake-uuid"}
    pg.ping.return_value   = True
    return pg


@pytest.fixture
def mock_temporal():
    pipeline = MagicMock()
    pipeline.predict.return_value = {"score": 0.85}
    return pipeline


@pytest.fixture
def service(mock_temporal, mock_redis, mock_pg):
    return PredictionService(
        temporal_pipeline = mock_temporal,
        redis             = mock_redis,
        pg                = mock_pg,
        cache_ttl         = 60,
    )


@pytest.fixture
def sample_request():
    return PredictionRequest(
        features      = {"temp": 22.5, "humidity": 0.4},
        model_name    = "weather_model",
        pipeline_type = "temporal",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Cache hit skips the pipeline
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_predict_cache_hit(service, mock_redis, sample_request):
    """Redis hit must return cached data and never call the ML pipeline."""
    mock_redis.get.return_value = {
        "prediction":    {"score": 0.99},
        "model_version": "v1",
    }

    response = await service.predict(sample_request)

    assert response.cached              is True
    assert response.prediction["score"] == 0.99
    service._temporal.predict.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Cache miss triggers pipeline + cache write + audit
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_predict_cache_miss_pipeline_success(
    service, mock_redis, mock_pg, sample_request
):
    """Cache miss must call pipeline, write result to Redis, write audit to PG."""
    mock_redis.get.return_value = None

    response = await service.predict(sample_request)

    assert response.cached     is False
    assert response.prediction == {"score": 0.85}
    mock_redis.set.assert_called_once()
    mock_pg.insert.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Redis GET failure is non-fatal
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_resilience_redis_get_failure_handled(service, mock_redis):
    """
    All redis.get calls raise.  The service must survive and return a prediction.

    Note: prediction_service.py has a bare redis.get() for version resolution
    inside _dispatch_pipeline that is wrapped in try/except — that is what
    allows the service to fall back to model_version='latest' and continue.
    """
    mock_redis.get.side_effect = Exception("Connection Refused")

    request = PredictionRequest(
        features      = {"a": 1.0},
        model_name    = "m",
        pipeline_type = "temporal",
    )
    response = await service.predict(request)

    assert response.prediction == {"score": 0.85}
    assert response.cached     is False


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Pipeline timeout raises TimeoutError
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_timeout_handling(service, sample_request):
    """Slow models must surface as asyncio.TimeoutError."""
    with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
        with pytest.raises(asyncio.TimeoutError):
            await service.predict(sample_request)


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Sensitivity analysis happy path
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sensitivity_analysis_happy_path(service):
    """
    Use patch.object(_ps_module, 'SensitivityAnalyzer') instead of a string.

    patch('prediction_service.SensitivityAnalyzer') can miss when pytest loads
    the service under both 'prediction_service' and 'service.prediction_service',
    producing two module objects.  The string-based patch targets whichever
    object the string resolves to, which may not be the one the service code
    reads.  patch.object always targets the exact module object we hold,
    guaranteeing the replacement is seen by the running service.
    """
    mock_result = MagicMock()
    mock_result.baseline_prediction = 0.123
    mock_result.feature_rankings    = [
        MagicMock(feature="temp",     sensitivity_score=0.9),
        MagicMock(feature="humidity", sensitivity_score=0.4),
    ]

    mock_model = MagicMock()
    mock_model.predict.return_value = [0.123]

    with patch.object(_ps_module, "SensitivityAnalyzer") as MockAnalyzer:
        MockAnalyzer.return_value.analyze.return_value = mock_result

        request = PredictionRequest(
            features      = {"temp": 22.5, "humidity": 0.4},
            model_name    = "weather_model",
            pipeline_type = "temporal",
        )
        result = await service.analyze_sensitivity(request, mock_model)

    assert result.baseline_prediction         == 0.123
    assert result.feature_rankings[0].feature == "temp"
    MockAnalyzer.assert_called_once_with(model=mock_model)
    MockAnalyzer.return_value.analyze.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — Health check reports correct per-backend state
# ─────────────────────────────────────────────────────────────────────────────

def test_health_check_logic(service, mock_redis, mock_pg):
    """Each backend is reported independently."""
    mock_redis.ping.return_value = True
    mock_pg.ping.return_value    = False

    status = service.health()

    assert status["redis"]    is True
    assert status["postgres"] is False