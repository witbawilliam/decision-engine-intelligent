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

TIMEOUTS = {
    "model_inference": 10.0,
    "cache_io":        0.5,
    "explanations":    30.0,   
}


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

   
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        set_trace_id(request.trace_id)
        start = time.perf_counter()

        with RequestContext(trace_id=request.trace_id):

            cache_key = self._make_cache_key(request.features, request.model_name)
            cached    = await self._cache_get(cache_key)
            if cached:
                # If explanations were requested but not in the root cache entry, 
                # we can optionally fetch them, but for now fallback smoothly
                return self._build_response(request, cached, start, cached=True)

            model, version = self._get_model(request.model_name)
            X              = pl.DataFrame([request.features])
            
            # Keep calculations concurrent where possible
            risk_score     = await self._risk_score(request.model_name, X)
            prediction     = await self._predict(model, X)

            result: Dict[str, Any] = {
                "prediction":    self._extract(prediction),
                "model_version": str(version),
                "risk_score":    risk_score,
                "explanations":  None,
            }

            # Run explanations BEFORE setting cache if requested inline
            if request.include_explanations:
                result["explanations"] = await self._run_explanations(
                    model, request, X
                )

            # Fire off background I/O tasks completely safely
            asyncio.create_task(self._cache_set(cache_key, result))
            asyncio.create_task(self._audit(request, result, start))

            return self._build_response(request, result, start)


    def _get_model(self, model_name: str):
        if model_name not in self._model_cache:
            model = self.registry.load(
                model_name=model_name,
                stage="production"
            )

            metadata = self.registry.list_versions(
                model_name=model_name,
                stage="production"
            )[0]

            version = metadata["version"]
            self._model_cache[model_name] = (model, version)

        return self._model_cache[model_name]


    async def _predict(self, model: Any, X: pl.DataFrame):
        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, model.predict, X),
            timeout=TIMEOUTS["model_inference"],
        )


    async def _risk_score(self, model_name: str, X: pl.DataFrame) -> float:
        # Offload fitting the guard sample from S3/DB to an executor thread 
        # to prevent blocking if a model is accessed for the first time.
        if model_name not in self._guard_cache:
            loop = asyncio.get_running_loop()
            def _init_guard():
                guard = ManifoldGuard()
                training = self.registry.load_training_sample(model_name)
                guard.fit(training)
                return guard
            
            self._guard_cache[model_name] = await loop.run_in_executor(None, _init_guard)
            
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._guard_cache[model_name].get_risk_score, X)

    

    async def _run_explanations(
        self,
        model,
        request: PredictionRequest,
        X: pl.DataFrame,
    ) -> Optional[Dict[str, Any]]:
        try:
            loop         = asyncio.get_running_loop()
            
            # Load training samples inside executor thread
            training     = await loop.run_in_executor(None, self.registry.load_training_sample, request.model_name)
            analyzer     = SensitivityAnalyzer(model)
            cf_orch      = CounterfactualOrchestrator(model, training)
            guard        = self._guard_cache.get(request.model_name)

            # Build task tracking arrays structurally 
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
                    None,
                    cf_orch.explain_how_to_hit_target,
                    X,
                    float(request.target_goal),
                    request.lever_col,
                    bounds,
                ))

            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=TIMEOUTS["explanations"],
            )

            # 🔧 FIX: Safely unpack by popping list indexes instead of hardcoded assignment
            sens_result = results[0]
            cf_result   = results[1] if cf_enabled else None

            sensitivity_payload = None
            if isinstance(sens_result, Exception):
                logger.warning("sensitivity_failed", extra={"error": str(sens_result)})
            else:
                sensitivity_payload = sens_result.model_dump() if hasattr(sens_result, "model_dump") else sens_result

            counterfactual_payload = None
            if cf_enabled and cf_result is not None:
                if isinstance(cf_result, Exception):
                    logger.warning("counterfactual_failed", extra={"error": str(cf_result)})
                else:
                    counterfactual_payload = cf_result.model_dump() if hasattr(cf_result, "model_dump") else cf_result

            manifold_payload = None
            if guard is not None:
                try:
                    manifold_payload = {
                        "risk_score":       guard.get_risk_score(X),
                        "feature_count":    guard._feature_count,
                        "numeric_columns":  guard._numeric_columns,
                    }
                except Exception as e:
                    logger.warning("manifold_detail_failed", extra={"error": str(e)})

            payload = {
                "sensitivity":    sensitivity_payload,
                "counterfactual": counterfactual_payload,
                "manifold":       manifold_payload,
            }

            asyncio.create_task(
                self._cache_set(f"exp:{request.request_id}", payload)
            )
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
            val  = await asyncio.wait_for(
                loop.run_in_executor(None, self.redis.get, key),
                timeout=TIMEOUTS["cache_io"],
            )
            return json.loads(val) if val else None
        except Exception:
            return None


    async def _cache_set(self, key: str, value: Dict) -> None:
        try:
            serialised = json.dumps(value, default=str)
            loop       = asyncio.get_running_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, self.redis.setex, key, 300, serialised),
                timeout=TIMEOUTS["cache_io"],
            )
        except Exception as e:
            logger.warning("cache_set_failed", extra={"error": str(e)})

    

    async def _audit(self, request: PredictionRequest, result: Dict, start: float) -> None:
        try:
            # 🔧 FIX: Execute the synchronous Postgres client call on an executor thread 
            # to prevent blocking the async loop worker
            loop = asyncio.get_running_loop()
            record = {
                "request_id": request.request_id,
                "model_name": request.model_name,
                "prediction": str(result["prediction"]),
                "latency_ms": (time.perf_counter() - start) * 1000,
                "timestamp":  datetime.now(timezone.utc),
            }
            await loop.run_in_executor(None, self.pg.insert, "prediction_audit", record)
        except Exception as e:
            logger.error("audit_failed", extra={"error": str(e)})


    def health(self) -> Dict[str, bool]:
        out = {"redis": False, "postgres": False, "registry": False}
        for name, fn in [
            ("redis",    lambda: self.redis.ping()),
            ("postgres", lambda: self.pg.ping()),
            ("registry", lambda: self.registry.ping()),
        ]:
            try:
                fn(); out[name] = True
            except Exception as e:
                logger.warning(f"health_{name}_failed", extra={"error": str(e)})
        return out


    def _make_cache_key(self, features: Dict, model_name: str) -> str:
        raw = json.dumps(features, sort_keys=True, default=str)
        return f"{model_name}:{hashlib.sha256(raw.encode()).hexdigest()}"

    def _extract(self, prediction: Any) -> Any:
        return prediction[0] if hasattr(prediction, "__len__") else prediction

    def _build_response(self, request, data, start, cached=False) -> PredictionResponse:
        return PredictionResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            prediction=data.get("prediction"),
            model_version=data.get("model_version", "unknown"),
            risk_score=data.get("risk_score", 0.0),
            latency_ms=(time.perf_counter() - start) * 1000,
            cached=cached,
            explanations=data.get("explanations"),
            trace_id=request.trace_id,
        )