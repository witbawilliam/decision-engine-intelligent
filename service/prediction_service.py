from __future__ import annotations

import asyncio
import json
import time
import uuid
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import polars as pl

from core.models.model_registry import ModelRegistry
from core.decision_engine.sensitivity_analysis import SensitivityAnalyzer
from core.decision_engine.counterfactuals import CounterfactualOrchestrator
from core.decision_engine.manifold_guard import ManifoldGuard

from storage.redis_client import RedisClient
from storage.postgres_client import PostgresClient
from monitoring.logging_config import get_logger, RequestContext, set_trace_id


logger = get_logger(__name__, component="prediction_service")


# -------------------------
# CONFIG
# -------------------------
TIMEOUTS = {
    "model_inference": 10.0,
    "cache_io": 0.5,
    "explanations": 10.0,
}


# -------------------------
# DATA CONTRACTS
# -------------------------
@dataclass
class PredictionRequest:
    features: Dict[str, Any]
    model_name: str
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    include_explanations: bool = False


@dataclass
class PredictionResponse:
    request_id: str
    model_name: str
    prediction: Any
    model_version: str
    risk_score: float
    latency_ms: float
    cached: bool = False
    explanations: Optional[Dict[str, Any]] = None
    trace_id: str = ""


# -------------------------
# SERVICE
# -------------------------
class PredictionService:

    def __init__(
        self,
        registry: ModelRegistry,
        redis: RedisClient,
        pg: PostgresClient,
    ):
        self.registry = registry
        self.redis = redis
        self.pg = pg

        # in-memory caches (VERY important)
        self._model_cache: Dict[str, Any] = {}
        self._guard_cache: Dict[str, ManifoldGuard] = {}

    # -------------------------
    # MAIN ENTRY
    # -------------------------
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        set_trace_id(request.trace_id)

        start = time.perf_counter()

        with RequestContext(trace_id=request.trace_id):

            # 1. cache lookup
            cache_key = self._make_cache_key(request.features, request.model_name)

            cached = await self._cache_get(cache_key)
            if cached:
                return self._build_response(request, cached, start, cached=True)

            # 2. load model (cached in memory)
            model, version = self._get_model(request.model_name)

            # 3. transform input
            X = pl.DataFrame([request.features])

            # 4. risk / OOD check
            risk_score = await self._risk_score(request.model_name, X)

            # 5. inference (fast path)
            prediction = await self._predict(model, X)

            result = {
                "prediction": self._extract(prediction),
                "model_version": str(version),
                "risk_score": risk_score,
            }

            # 6. async side effects (DO NOT block)
            asyncio.create_task(self._cache_set(cache_key, result))
            asyncio.create_task(self._audit(request, result, start))

            # 7. optional explanations (slow path)
            if request.include_explanations:
                asyncio.create_task(
                    self._run_explanations(model, request, X)
                )

            return self._build_response(request, result, start)

    # -------------------------
    # MODEL LOADING (CACHED)
    # -------------------------
    def _get_model(self, model_name: str):
        if model_name in self._model_cache:
            return self._model_cache[model_name]

        model = self.registry.load_model(model_name)
        version = self.registry.get_latest_version(model_name)

        self._model_cache[model_name] = (model, version)
        return model, version

    # -------------------------
    # PREDICTION
    # -------------------------
    async def _predict(self, model: Any, X: pl.DataFrame):
        loop = asyncio.get_running_loop()

        return await asyncio.wait_for(
            loop.run_in_executor(None, model.predict, X),
            timeout=TIMEOUTS["model_inference"]
        )

    # -------------------------
    # RISK / MANIFOLD
    # -------------------------
    async def _risk_score(self, model_name: str, X: pl.DataFrame) -> float:
        if model_name not in self._guard_cache:
            guard = ManifoldGuard()
            training = self.registry.load_training_sample(model_name)
            guard.fit(training)
            self._guard_cache[model_name] = guard

        return self._guard_cache[model_name].get_risk_score(X)

    # -------------------------
    # EXPLANATIONS (ASYNC ONLY)
    # -------------------------
    async def _run_explanations(self, model, request: PredictionRequest, X):
        try:
            analyzer = SensitivityAnalyzer(model)
            training = self.registry.load_training_sample(request.model_name)
            cf = CounterfactualOrchestrator(model, training)

            loop = asyncio.get_running_loop()

            sens_task = loop.run_in_executor(None, analyzer.analyze, X)
            cf_task = loop.run_in_executor(None, cf.explain, X)

            sens, counterfactual = await asyncio.gather(sens_task, cf_task)

            payload = {
                "sensitivity": getattr(sens, "model_dump", lambda: sens)(),
                "counterfactual": getattr(counterfactual, "model_dump", lambda: counterfactual)(),
            }

            await self._cache_set(f"exp:{request.request_id}", payload)

        except Exception as e:
            logger.error("explanation_failed", extra={"error": str(e)})

    # -------------------------
    # CACHE
    # -------------------------
    async def _cache_get(self, key: str):
        try:
            loop = asyncio.get_running_loop()
            val = await loop.run_in_executor(None, self.redis.get, key)
            return json.loads(val) if val else None
        except:
            return None

    async def _cache_set(self, key: str, value: Dict):
        try:
            loop = asyncio.get_running_loop()
            self.redis.setex(key, 300, json.dumps(value))
        except Exception as e:
            logger.warning("cache_failed", extra={"error": str(e)})

    # -------------------------
    # AUDIT
    # -------------------------
    async def _audit(self, request, result, start):
        try:
            latency = (time.perf_counter() - start) * 1000

            record = {
                "request_id": request.request_id,
                "model_name": request.model_name,
                "prediction": str(result["prediction"]),
                "latency_ms": latency,
                "timestamp": datetime.now(timezone.utc),
            }

            self.pg.insert("prediction_audit", record)

        except Exception as e:
            logger.error("audit_failed", extra={"error": str(e)})

    # -------------------------
    # HELPERS
    # -------------------------
    def _make_cache_key(self, features, model_name):
        raw = json.dumps(features, sort_keys=True)
        return f"{model_name}:{hashlib.sha256(raw.encode()).hexdigest()}"

    def _extract(self, prediction):
        return prediction[0] if hasattr(prediction, "__len__") else prediction

    def _build_response(self, request, data, start, cached=False):
        return PredictionResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            prediction=data["prediction"],
            model_version=data["model_version"],
            risk_score=data["risk_score"],
            latency_ms=(time.perf_counter() - start) * 1000,
            cached=cached,
            trace_id=request.trace_id,
        )