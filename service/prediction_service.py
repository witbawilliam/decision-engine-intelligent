from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Type

import polars as pl

from core.pipelines.temporal_pipeline import TemporalPipeline
from core.pipelines.tabular_pipeline import TabularPipeline

from storage.redis_client import RedisClient
from storage.postgres_client import PostgresClient

from monitoring.logging_config import get_logger, RequestContext, set_trace_id, get_trace_id
from monitoring.metrics import MetricsRegistry, track_training_latency

from core.decision_engine.sensitivity_analysis import SensitivityAnalyzer, SensitivityResult
from core.decision_engine.counterfactuals import CounterfactualOrchestrator, CounterfactualResult
from core.decision_engine.manifold_guard import ManifoldGuard


logger = get_logger(__name__, component="inference_engine")



TIMEOUT_PIPELINE_PREDICT: float = 30.0   # ML model inference
TIMEOUT_REDIS_GET:        float =  1.0   # Cache read
TIMEOUT_REDIS_SET:        float =  1.0   # Cache write
TIMEOUT_PG_WRITE:         float =  5.0   # Audit log persistence
TIMEOUT_SENSITIVITY:      float = 10.0   # Sensitivity sweep
TIMEOUT_COUNTERFACTUAL:   float = 15.0   # Binary search optimisation



_CACHE_TTL_SECONDS = 300   # 5-minute prediction cache


def _cache_key(model_name: str, features_hash: str) -> str:
    return f"prediction:{model_name}:{features_hash}"


def _request_lock_key(request_id: str) -> str:
    return f"inflight:{request_id}"


def _features_hash(features: Dict[str, Any]) -> str:
    """Stable hash of the feature dict for cache keying."""
    return str(hash(json.dumps(features, sort_keys=True, default=str)))



@dataclass
class PredictionRequest:
    features:    Dict[str, Any]
    model_name:  str
    request_id:  str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id:    str = field(default_factory=lambda: str(uuid.uuid4()))
    pipeline_type: str = "temporal"   # "temporal" | "tabular"


@dataclass
class PredictionResponse:
    request_id:     str
    model_name:     str
    model_version:  str
    prediction:     Any
    latency_ms:     float
    trace_id:       str
    cached:         bool = False
    timestamp:      str  = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)




class PredictionService:
    

    def __init__(
        self,
        *,
        temporal_pipeline: Optional[TemporalPipeline] = None,
        tabular_pipeline:  Optional[TabularPipeline]  = None,
        redis:             Optional[Type[RedisClient]] = None,
        pg:                Optional[Type[PostgresClient]] = None,
        training_data:     Optional[pl.DataFrame] = None,   # for ManifoldGuard + Counterfactuals
        cache_ttl:         int   = _CACHE_TTL_SECONDS,
        audit_table:       str   = "prediction_audit_log",
    ) -> None:
        
        self._temporal = temporal_pipeline
        self._tabular  = tabular_pipeline

        
        self._redis = redis or RedisClient
        self._pg    = pg    or PostgresClient

        # Analysis modules (stateful, built once) 
        self._manifold_guard: Optional[ManifoldGuard] = None
        self._sensitivity:    Optional[SensitivityAnalyzer] = None

        if training_data is not None and not training_data.is_empty():
            self._manifold_guard = ManifoldGuard()
            self._manifold_guard.fit(training_data)
            logger.info(
                "manifold_guard_ready",
                extra={"event": "manifold_guard_fitted",
                       "rows": training_data.height,
                       "trace_id": "startup"},
            )

        
        self._cache_ttl   = cache_ttl
        self._audit_table = audit_table

    # Public async API 

    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Execute a prediction with full observability, caching, and timeouts.

        Flow
        
         Set trace_id on the current thread (propagates into all log records).
         Check Redis cache — return immediately on a hit.
         Acquire a per-request Redis lock to prevent duplicate in-flight work.
         Route to the correct pipeline (temporal / tabular).
         Wrap pipeline call in asyncio.wait_for() so it cannot hang.
         Cache the result in Redis (non-blocking, failure is non-fatal).
         Persist audit record to PostgreSQL (non-blocking, failure is non-fatal).
         Record latency metric via MetricsRegistry.
         Return structured PredictionResponse.
        """
        set_trace_id(request.trace_id)
        start = time.perf_counter()

        with RequestContext(trace_id=request.trace_id):
            logger.info(
                "prediction_request_received",
                extra={
                    "event":       "prediction_request_received",
                    "request_id":  request.request_id,
                    "model_name":  request.model_name,
                    "pipeline":    request.pipeline_type,
                    "timestamp":   datetime.now(timezone.utc).isoformat(),
                    "trace_id":    request.trace_id,
                },
            )

            try:
                #  Cache lookup 
                features_hash = _features_hash(request.features)
                cached_result = await self._cache_get(
                    _cache_key(request.model_name, features_hash)
                )

                if cached_result is not None:
                    latency_ms = (time.perf_counter() - start) * 1000
                    MetricsRegistry.increment(
                        "prediction_cache_hit",
                        tags={"model": request.model_name},
                    )
                    logger.info(
                        "prediction_cache_hit",
                        extra={
                            "event":      "prediction_cache_hit",
                            "request_id": request.request_id,
                            "latency_ms": round(latency_ms, 2),
                            "timestamp":  datetime.now(timezone.utc).isoformat(),
                            "trace_id":   request.trace_id,
                        },
                    )
                    return PredictionResponse(
                        request_id    = request.request_id,
                        model_name    = request.model_name,
                        model_version = cached_result.get("model_version", "unknown"),
                        prediction    = cached_result["prediction"],
                        latency_ms    = latency_ms,
                        trace_id      = request.trace_id,
                        cached        = True,
                    )

                #  In-flight deduplication lock 
                lock_key = _request_lock_key(request.request_id)
                with self._redis.lock(lock_key, timeout=TIMEOUT_PIPELINE_PREDICT + 5):

                    
                    prediction, model_version = await self._dispatch_pipeline(request)

                #  Cache the result 
                await self._cache_set(
                    key=_cache_key(request.model_name, features_hash),
                    value={"prediction": prediction, "model_version": model_version},
                )

                latency_ms = (time.perf_counter() - start) * 1000

                #  Audit
                await self._audit(request, prediction, model_version, latency_ms)

                
                MetricsRegistry.increment(
                    "prediction_success",
                    tags={"model": request.model_name, "pipeline": request.pipeline_type},
                )
                track_training_latency(
                    task_id  = request.request_id,
                    status   = "success",
                    pipeline = request.pipeline_type,
                )

                logger.info(
                    "prediction_success",
                    extra={
                        "event":         "prediction_success",
                        "request_id":    request.request_id,
                        "model_name":    request.model_name,
                        "model_version": model_version,
                        "latency_ms":    round(latency_ms, 2),
                        "timestamp":     datetime.now(timezone.utc).isoformat(),
                        "trace_id":      request.trace_id,
                    },
                )

                return PredictionResponse(
                    request_id    = request.request_id,
                    model_name    = request.model_name,
                    model_version = model_version,
                    prediction    = prediction,
                    latency_ms    = latency_ms,
                    trace_id      = request.trace_id,
                    cached        = False,
                )

            except asyncio.TimeoutError as exc:
                latency_ms = (time.perf_counter() - start) * 1000
                self._log_error("prediction_timeout", request, exc, latency_ms)
                MetricsRegistry.increment("prediction_timeout", tags={"model": request.model_name})
                raise

            except Exception as exc:
                latency_ms = (time.perf_counter() - start) * 1000
                self._log_error("prediction_failed", request, exc, latency_ms)
                MetricsRegistry.increment("prediction_error", tags={"model": request.model_name})
                track_training_latency(
                    task_id  = request.request_id,
                    status   = "error",
                    pipeline = request.pipeline_type,
                )
                raise

    

    async def analyze_sensitivity(
        self,
        request: PredictionRequest,
        pipeline_model: Any,
    ) -> SensitivityResult:
        """
        Run a per-feature sensitivity sweep on a single prediction row.
        Wraps SensitivityAnalyzer with a timeout and structured logging.
        """
        set_trace_id(request.trace_id)
        row = pl.DataFrame(request.features)

        analyzer = SensitivityAnalyzer(model=pipeline_model)

        logger.info(
            "sensitivity_analysis_start",
            extra={
                "event":      "sensitivity_analysis_start",
                "request_id": request.request_id,
                "timestamp":  datetime.now(timezone.utc).isoformat(),
                "trace_id":   request.trace_id,
            },
        )

        try:
            result: SensitivityResult = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, analyzer.analyze, row
                ),
                timeout=TIMEOUT_SENSITIVITY,
            )
        except asyncio.TimeoutError:
            logger.error(
                "sensitivity_analysis_timeout",
                extra={
                    "event":      "sensitivity_analysis_timeout",
                    "request_id": request.request_id,
                    "timestamp":  datetime.now(timezone.utc).isoformat(),
                    "trace_id":   request.trace_id,
                },
            )
            raise

        logger.info(
            "sensitivity_analysis_complete",
            extra={
                "event":             "sensitivity_analysis_complete",
                "request_id":        request.request_id,
                "top_feature":       result.feature_rankings[0].feature if result.feature_rankings else "none",
                "baseline_prediction": result.baseline_prediction,
                "timestamp":         datetime.now(timezone.utc).isoformat(),
                "trace_id":          request.trace_id,
            },
        )
        return result

    

    async def explain_counterfactual(
        self,
        request:       PredictionRequest,
        pipeline_model: Any,
        training_data:  pl.DataFrame,
        lever_col:      str,
        target_goal:    float,
        bounds:         tuple,
    ) -> CounterfactualResult:
        """
        Run a counterfactual optimisation: what value of `lever_col` achieves
        `target_goal`? Wraps CounterfactualOrchestrator with timeout + logging.
        """
        set_trace_id(request.trace_id)
        row = pl.DataFrame(request.features)

        orchestrator = CounterfactualOrchestrator(
            model         = pipeline_model,
            training_data = training_data,
        )

        logger.info(
            "counterfactual_start",
            extra={
                "event":       "counterfactual_start",
                "request_id":  request.request_id,
                "lever_col":   lever_col,
                "target_goal": target_goal,
                "timestamp":   datetime.now(timezone.utc).isoformat(),
                "trace_id":    request.trace_id,
            },
        )

        try:
            result: CounterfactualResult = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: orchestrator.explain_how_to_hit_target(
                        original_row = row,
                        target_goal  = target_goal,
                        lever_col    = lever_col,
                        bounds       = bounds,
                        strict       = False,
                    ),
                ),
                timeout=TIMEOUT_COUNTERFACTUAL,
            )
        except asyncio.TimeoutError:
            logger.error(
                "counterfactual_timeout",
                extra={
                    "event":      "counterfactual_timeout",
                    "request_id": request.request_id,
                    "timestamp":  datetime.now(timezone.utc).isoformat(),
                    "trace_id":   request.trace_id,
                },
            )
            raise

        logger.info(
            "counterfactual_complete",
            extra={
                "event":               "counterfactual_complete",
                "request_id":          request.request_id,
                "status":              result.status,
                "risk_score":          result.risk_score,
                "achieved_prediction": result.achieved_prediction,
                "timestamp":           datetime.now(timezone.utc).isoformat(),
                "trace_id":            request.trace_id,
            },
        )
        return result

    

    def get_risk_score(self, request: PredictionRequest) -> float:
        """
        Return the ManifoldGuard out-of-distribution risk score for the request.
        Returns 0.0 if no training data was provided at construction time.
        """
        if self._manifold_guard is None:
            logger.warning(
                "manifold_guard_not_fitted",
                extra={
                    "event":     "manifold_guard_not_fitted",
                    "message":   "No training data supplied — returning risk_score=0.0",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "trace_id":  get_trace_id(),
                },
            )
            return 0.0

        row = pl.DataFrame(request.features)
        return self._manifold_guard.get_risk_score(row)

    

    def health(self) -> Dict[str, bool]:
        """Ping Redis and PostgreSQL. Used by load-balancer /health endpoints."""
        status = {
            "redis":    self._redis.ping(),
            "postgres": self._pg.ping(),
        }
        logger.info(
            "health_check",
            extra={
                "event":     "health_check",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trace_id":  get_trace_id(),
                **status,
            },
        )
        return status

    

    async def _dispatch_pipeline(
        self, request: PredictionRequest
    ) -> tuple[Any, str]:
        """
        Route to the correct pipeline based on request.pipeline_type,
        wrap in asyncio.wait_for(), and return (prediction, model_version).
        """
        input_df = pl.DataFrame(request.features)

        if request.pipeline_type == "temporal":
            if self._temporal is None:
                raise RuntimeError(
                    "TemporalPipeline not injected. "
                    "Pass temporal_pipeline= to PredictionService.__init__()."
                )
            pipeline = self._temporal
        elif request.pipeline_type == "tabular":
            if self._tabular is None:
                raise RuntimeError(
                    "TabularPipeline not injected. "
                    "Pass tabular_pipeline= to PredictionService.__init__()."
                )
            pipeline = self._tabular
        else:
            raise ValueError(
                f"Unknown pipeline_type '{request.pipeline_type}'. "
                "Expected 'temporal' or 'tabular'."
            )

        try:
            raw_result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, pipeline.predict, input_df
                ),
                timeout=TIMEOUT_PIPELINE_PREDICT,
            )
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"Pipeline predict() exceeded {TIMEOUT_PIPELINE_PREDICT}s timeout."
            )

        # Normalise result — pipelines may return an object with .predictions
        # or a raw array / scalar.
        if hasattr(raw_result, "predictions"):
            prediction = raw_result.predictions
        else:
            prediction = raw_result

        # Best-effort version resolution from Redis (non-blocking, non-fatal).
        version_key = f"model_version:{request.model_name}"
        try:
            model_version = self._redis.get(version_key) or "latest"
        except Exception:
            model_version = "latest"

        return prediction, str(model_version)

    async def _cache_get(self, key: str) -> Optional[Any]:
        """Non-blocking Redis GET with timeout. Returns None on miss or timeout."""
        try:
            return await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self._redis.get, key
                ),
                timeout=TIMEOUT_REDIS_GET,
            )
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning(
                "cache_get_failed",
                extra={
                    "event":     "cache_get_failed",
                    "key":       key,
                    "error":     str(exc),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "trace_id":  get_trace_id(),
                },
            )
            return None

    async def _cache_set(self, key: str, value: Any) -> None:
        """Non-blocking Redis SET with timeout. Failure is logged but non-fatal."""
        try:
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._redis.set(key, value, ttl=self._cache_ttl),
                ),
                timeout=TIMEOUT_REDIS_SET,
            )
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning(
                "cache_set_failed",
                extra={
                    "event":     "cache_set_failed",
                    "key":       key,
                    "error":     str(exc),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "trace_id":  get_trace_id(),
                },
            )

    async def _audit(
        self,
        request:       PredictionRequest,
        prediction:    Any,
        model_version: str,
        latency_ms:    float,
    ) -> None:
        """
        Persist a prediction audit record to PostgreSQL via PostgresClient.insert().
        Wrapped in asyncio.wait_for(); failure is logged but never propagates —
        a DB blip must not take down inference.
        """
        record = {
            "id":            str(uuid.uuid4()),
            "request_id":    request.request_id,
            "trace_id":      request.trace_id,
            "model_name":    request.model_name,
            "model_version": model_version,
            "pipeline_type": request.pipeline_type,
            "prediction":    json.dumps(prediction, default=str),
            "latency_ms":    round(latency_ms, 2),
            "recorded_at":   datetime.now(timezone.utc),
        }

        try:
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._pg.insert(
                        table     = self._audit_table,
                        data      = record,
                        returning = "id",
                    ),
                ),
                timeout=TIMEOUT_PG_WRITE,
            )
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning(
                "audit_write_failed",
                extra={
                    "event":      "audit_write_failed",
                    "request_id": request.request_id,
                    "error":      str(exc),
                    "timestamp":  datetime.now(timezone.utc).isoformat(),
                    "trace_id":   request.trace_id,
                },
            )

    def _log_error(
        self,
        event:      str,
        request:    PredictionRequest,
        exc:        Exception,
        latency_ms: float,
    ) -> None:
        """Emit a structured error log with all required observability fields."""
        logger.error(
            event,
            extra={
                "event":       event,
                "request_id":  request.request_id,
                "model_name":  request.model_name,
                "pipeline":    request.pipeline_type,
                "error_type":  type(exc).__name__,
                "error":       str(exc),
                "latency_ms":  round(latency_ms, 2),
                "timestamp":   datetime.now(timezone.utc).isoformat(),
                "trace_id":    request.trace_id,
            },
            exc_info=True,
        )