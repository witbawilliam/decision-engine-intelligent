from __future__ import annotations

import asyncio
import json
import time
import uuid
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import polars as pl

from core.models.model_registry import ModelRegistry
from core.decision_engine.sensitivity_analysis import SensitivityAnalyzer
from core.decision_engine.counterfactuals import CounterfactualOrchestrator
from core.decision_engine.manifold_guard import ManifoldGuard

from storage.redis_client import RedisClient
from storage.postgres_client import PostgresClient
from monitoring.logging_config import get_logger, RequestContext, set_trace_id

logger = get_logger(__name__, component="prediction_service")

TIMEOUTS = {
    "model_inference": 10.0,
    "cache_io":        0.5,
    "explanations":    30.0,   
}

def normalize_model_identifier(model_name: str) -> str:
    """
    Pipeline-agnostic utility to strip path configurations, directories, 
    and file extensions from raw tracking parameters cleanly.
    """
    normalized = model_name
    if "/" in model_name or ".parquet" in model_name:
        normalized = model_name.split("/")[-1]
        normalized = normalized.replace(".parquet_", "_")
        normalized = normalized.replace(".parquet", "")
    return normalized

@dataclass
class PredictionRequest:
    features:   Dict[str, Any]
    model_name: str
    request_id: str  = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id:   str  = field(default_factory=lambda: str(uuid.uuid4()))
    include_explanations: bool = False

    lever_col:   Optional[str]   = None   
    target_goal: Optional[float] = None   
    lever_min:   Optional[float] = None   
    lever_max:   Optional[float] = None   

@dataclass
class PredictionResponse:
    request_id:    str
    model_name:    str
    prediction:    Any
    model_version: str
    risk_score:    float
    latency_ms:    float
    cached:        bool                    = False
    explanations:  Optional[Dict[str, Any]] = None
    trace_id:      str                     = ""

class PredictionService:

    def __init__(self, registry: ModelRegistry, redis: RedisClient, pg: PostgresClient):
        self.registry = registry
        self.redis    = redis
        self.pg       = pg
        self._model_cache: Dict[str, Any]           = {}
        self._guard_cache: Dict[str, ManifoldGuard] = {}
        self._training_cache: Dict[str, pl.DataFrame] = {}  # In-memory background data cache

    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        set_trace_id(request.trace_id)
        start = time.perf_counter()

        clean_model_name = normalize_model_identifier(request.model_name)

        with RequestContext(trace_id=request.trace_id):
            cache_key = self._make_cache_key(request.features, clean_model_name)
            cached    = await self._cache_get(cache_key)
            if cached:
                return self._build_response(request, cached, start, clean_model_name, cached=True)

            model, version = await self._get_model(clean_model_name)
            X              = pl.DataFrame([request.features])
            
            risk_score     = await self._risk_score(clean_model_name, X)
            prediction     = await self._predict(model, X)

            result: Dict[str, Any] = {
                "prediction":    self._extract(prediction),
                "model_version": str(version),
                "risk_score":    risk_score,
                "explanations":  None,
            }

            if request.include_explanations:
                # Optimized: Pass the precomputed risk_score to save CPU cycles
                result["explanations"] = await self._run_explanations(
                    model, request, clean_model_name, X, precomputed_risk=risk_score
                )

            asyncio.create_task(self._cache_set(cache_key, result))
            asyncio.create_task(self._audit(clean_model_name, request, result, start))

            return self._build_response(request, result, start, clean_model_name)

    async def _get_model(self, clean_model_name: str):
        if clean_model_name not in self._model_cache:
            loop = asyncio.get_running_loop()
            
            def _load_sync():
                model = self.registry.load(model_name=clean_model_name, stage="production")
                metadata = self.registry.list_versions(model_name=clean_model_name, stage="production")[0]
                return model, metadata["version"]

            model, version = await loop.run_in_executor(None, _load_sync)
            self._model_cache[clean_model_name] = (model, version)

        return self._model_cache[clean_model_name]

    async def _get_training_sample(self, clean_model_name: str) -> pl.DataFrame:
        """Ensures background training dataframes are cached in RAM after the first pull."""
        if clean_model_name not in self._training_cache:
            loop = asyncio.get_running_loop()
            training_df = await loop.run_in_executor(
                None, self.registry.load_training_sample, clean_model_name
            )
            self._training_cache[clean_model_name] = training_df
        return self._training_cache[clean_model_name]

    async def _predict(self, model: Any, X: pl.DataFrame):
        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, model.predict, X),
            timeout=TIMEOUTS["model_inference"],
        )

    async def _risk_score(self, clean_model_name: str, X: pl.DataFrame) -> float:
        if clean_model_name not in self._guard_cache:
            # We fetch training data via our cached loader to bootstrap the ManifoldGuard
            training = await self._get_training_sample(clean_model_name)
            
            loop = asyncio.get_running_loop()
            def _init_guard():
                guard = ManifoldGuard()
                guard.fit(training)
                return guard
            
            self._guard_cache[clean_model_name] = await loop.run_in_executor(None, _init_guard)
            
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._guard_cache[clean_model_name].get_risk_score, X)

    async def _run_explanations(self, model, request: PredictionRequest, clean_model_name: str, X: pl.DataFrame, precomputed_risk: float) -> Optional[Dict[str, Any]]:
        try:
            loop         = asyncio.get_running_loop()
            
            # Optimized: Pull background sample from local storage cache memory
            training     = await self._get_training_sample(clean_model_name)
            
            analyzer     = SensitivityAnalyzer(model)
            cf_orch      = CounterfactualOrchestrator(model, training)
            guard        = self._guard_cache.get(clean_model_name)

            tasks = []
            tasks.append(loop.run_in_executor(None, analyzer.analyze, X))
            
            cf_enabled = (
                request.lever_col
                and request.target_goal is not None
                and request.lever_min   is not None
                and request.lever_max   is not None
                and request.lever_col in X.columns
            )

            if cf_enabled:
                bounds = (float(request.lever_min), float(request.lever_max))
                tasks.append(loop.run_in_executor(
                    None, cf_orch.explain_how_to_hit_target, X,
                    float(request.target_goal), request.lever_col, bounds
                ))

            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=TIMEOUTS["explanations"],
            )

            sens_result = results[0]
            cf_result   = results[1] if cf_enabled else None

            sensitivity_payload = None
            if isinstance(sens_result, Exception):
                logger.warning("sensitivity_failed", extra={"error": str(sens_result)})
            else:
                # Use model_dump(mode="json") if available to handle enum/date conversions cleanly
                sensitivity_payload = sens_result.model_dump(mode="json") if hasattr(sens_result, "model_dump") else sens_result

            counterfactual_payload = None
            if cf_enabled and cf_result is not None:
                if isinstance(cf_result, Exception):
                    logger.warning("counterfactual_failed", extra={"error": str(cf_result)})
                else:
                    if hasattr(cf_result, "model_dump"):
                        counterfactual_payload = cf_result.model_dump(mode="json")
                    else:
                        counterfactual_payload = cf_result

            manifold_payload = None
            if guard is not None:
                manifold_payload = {
                    "risk_score":       precomputed_risk,  # Reused precomputed float variable safely
                    "feature_count":    int(guard._feature_count),
                    "numeric_columns":  list(guard._numeric_columns),
                }

            payload = {
                "sensitivity":    sensitivity_payload,
                "counterfactual": counterfactual_payload,
                "manifold":       manifold_payload,
            }

            asyncio.create_task(self._cache_set(f"exp:{request.request_id}", payload))
            return payload

        except asyncio.TimeoutError:
            logger.warning("explanation_timeout", extra={"request_id": request.request_id})
            return None
        except Exception as e:
            logger.error("explanation_failed", extra={"error": str(e)})
            return None

    async def _cache_get(self, key: str) -> Optional[Dict]:
        try:
            loop = asyncio.get_running_loop()
            val  = await asyncio.wait_for(loop.run_in_executor(None, self.redis.get, key), timeout=TIMEOUTS["cache_io"])
            return json.loads(val) if val else None
        except Exception:
            return None

    async def _cache_set(self, key: str, value: Dict) -> None:
        try:
            serialised = json.dumps(value, default=str)
            loop       = asyncio.get_running_loop()
            await asyncio.wait_for(loop.run_in_executor(None, self.redis.setex, key, 300, serialised), timeout=TIMEOUTS["cache_io"])
        except Exception as e:
            logger.warning("cache_set_failed", extra={"error": str(e)})

    async def _audit(self, clean_model_name: str, request: PredictionRequest, result: Dict, start: float) -> None:
        try:
            loop = asyncio.get_running_loop()
            record = {
                "request_id": request.request_id,
                "model_name": clean_model_name,
                "prediction": str(result["prediction"]),
                "latency_ms": (time.perf_counter() - start) * 1000,
                "timestamp":  datetime.now(timezone.utc),
            }
            await loop.run_in_executor(None, self.pg.insert, "prediction_audit", record)
        except Exception as e:
            logger.error("audit_failed", extra={"error": str(e)})

    def _make_cache_key(self, features: Dict, model_name: str) -> str:
        raw = json.dumps(features, sort_keys=True, default=str)
        return f"{model_name}:{hashlib.sha256(raw.encode()).hexdigest()}"

    def _extract(self, prediction: Any) -> Any:
        """Forces predictions into JSON-serializable native types."""
        if hasattr(prediction, "to_numpy"):
            prediction = prediction.to_numpy()
        if hasattr(prediction, "__len__") and len(prediction) > 0:
            val = prediction[0]
        else:
            val = prediction
        # Strip numpy wrappers down to primitive float/int
        return val.item() if hasattr(val, "item") else val

    def _build_response(self, request, data, start, clean_model_name: str, cached=False) -> PredictionResponse:
        return PredictionResponse(
            request_id=request.request_id,
            model_name=clean_model_name,
            prediction=data.get("prediction"),
            model_version=data.get("model_version", "unknown"),
            risk_score=float(data.get("risk_score", 0.0)),
            latency_ms=(time.perf_counter() - start) * 1000,
            cached=cached,
            explanations=data.get("explanations"),
            trace_id=request.trace_id,
        )