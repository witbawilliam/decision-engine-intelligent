from __future__ import annotations

import logging
import time
import uuid
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import polars as pl

from core.models.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

_CACHE_VERSION_PREFIX = "forecast:prod_version:"


class ForecastServiceError(Exception):
    """Raised for any recoverable failure in the inference pipeline, with context."""


@dataclass
class ForecastServiceResult:
    product_id:    str
    model_name:    str
    model_version: str
    periods:       int
    forecast:      pd.DataFrame
    generated_at:  str


@dataclass
class PredictionRequest:
    model_name: str
    features: Dict[str, Any]
    include_explanations: bool = False
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    lever_col: Optional[str] = None
    target_goal: Optional[float] = None
    lever_min: Optional[float] = None
    lever_max: Optional[float] = None


@dataclass
class PredictionResponse:
    request_id: str
    model_name: str
    model_version: str
    prediction: Any
    latency_ms: float
    generated_at: str


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively replace NaN/Infinity with None so json.dumps doesn't
    produce invalid JSON (Postgres JSONB rejects NaN/Infinity outright)."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(x) for x in obj]
    return obj


class PredictionService:
    """
    Steps: determine model name -> load production model -> generate future
    dataframe -> run prediction -> extract future-only rows -> validate -> return.

    predict() adapts the generic InferencePayloadSchema (features dict) used
    by routes_inference.py onto the forecasting-specific run() flow, since
    forecasting needs product_id + periods, not an arbitrary feature vector.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        redis=None,
        pg=None,
        default_forecast_horizon: int = 30,
        default_forecast_freq: str = "D",
        model_name_prefix: str = "prophet_product_",
        cache_ttl_seconds: int = 3600,
    ):
        self.registry = registry
        self.redis = redis
        self.pg = pg
        self.default_forecast_horizon = default_forecast_horizon
        self.default_forecast_freq = default_forecast_freq
        self.model_name_prefix = model_name_prefix
        self.cache_ttl_seconds = cache_ttl_seconds
        # model_name -> (model, metadata, cached_version, cached_at)
        self._model_cache: Dict[str, tuple] = {}

    # ---------- Public API expected by routes_inference.py ----------

    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        start = time.perf_counter()
        request_id = request.request_id or str(uuid.uuid4())

        product_id = request.features.get("product_id")
        if not product_id:
            raise KeyError("features.product_id is required for forecast prediction.")
        periods = request.features.get("periods")

        result = self.run(product_id=product_id, periods=periods)

        latency_ms = (time.perf_counter() - start) * 1000
        response = PredictionResponse(
            request_id=request_id,
            model_name=result.model_name,
            model_version=result.model_version,
            prediction=result.forecast.to_dict(orient="records"),
            latency_ms=round(latency_ms, 2),
            generated_at=result.generated_at,
        )

        if self.pg is not None:
            self._save_prediction_audit(request_id, result, request, latency_ms)

        return response

    async def get_model_predictions(
        self, model_name: str, version: Optional[str], limit: int, offset: int
    ) -> List[Dict[str, Any]]:
        if self.pg is None:
            raise ValueError("Postgres client not configured for this service.")
        if version:
            rows = self.pg.query(
                "SELECT * FROM forecast_evaluations WHERE model_name = %s AND model_version = %s "
                "ORDER BY created_at DESC LIMIT %s OFFSET %s",
                (model_name, version, limit, offset),
            )
        else:
            rows = self.pg.query(
                "SELECT * FROM forecast_evaluations WHERE model_name = %s "
                "ORDER BY created_at DESC LIMIT %s OFFSET %s",
                (model_name, limit, offset),
            )
        if not rows:
            raise ValueError(f"No predictions found for model_name='{model_name}'.")
        return rows

    async def get_prediction_audit(
        self, model_name: str, limit: int, offset: int
    ) -> List[Dict[str, Any]]:
        if self.pg is None:
            raise ValueError("Postgres client not configured for this service.")
        rows = self.pg.query(
            "SELECT * FROM prediction_audit WHERE model_name = %s "
            "ORDER BY timestamp DESC LIMIT %s OFFSET %s",
            (model_name, limit, offset),
        )
        if not rows:
            raise ValueError(f"No audit records found for model_name='{model_name}'.")
        for row in rows:
            if isinstance(row.get("features"), str):
                row["features"] = json.loads(row["features"])
            if isinstance(row.get("prediction"), str):
                row["prediction"] = json.loads(row["prediction"])
        return rows

    def _save_prediction_audit(
        self, request_id: str, result: ForecastServiceResult,
        request: PredictionRequest, latency_ms: float,
    ) -> None:
        try:
            forecast_records = result.forecast.to_dict(orient="records")
            clean_features = _sanitize_for_json(request.features)
            clean_prediction = _sanitize_for_json(forecast_records)

            self.pg.execute(
                """
                INSERT INTO prediction_audit
                    (request_id, model_name, model_version, features, prediction, latency_ms)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    request_id,
                    result.model_name,
                    result.model_version,
                    json.dumps(clean_features),
                    json.dumps(clean_prediction),
                    latency_ms,
                ),
            )
        except Exception:
            logger.exception("Failed to write prediction_audit for request_id=%s", request_id)

    # ---------- Cache management ----------

    def _get_current_production_version(self, model_name: str) -> Optional[int]:
        """Reads the promotion pointer from Redis, if configured. Returns
        None on any Redis error or absent key -- caller falls back to the
        local TTL cache in that case."""
        if self.redis is None:
            return None
        try:
            raw = self.redis.get(f"{_CACHE_VERSION_PREFIX}{model_name}")
            return int(raw) if raw is not None else None
        except Exception:
            logger.warning("Redis read failed for '%s' — falling back to TTL cache only.", model_name)
            return None

    def invalidate_cache(self, model_name: str) -> None:
        """Call this after promote_to_production() succeeds so every worker
        process picks up the new version on its next request instead of
        waiting out the TTL."""
        self._model_cache.pop(model_name, None)

    def _load_production_model(self, model_name: str):
        current_version = self._get_current_production_version(model_name)
        cached = self._model_cache.get(model_name)

        if cached is not None:
            model, preprocessor, metadata, cached_version, cached_at = cached
            is_fresh = (time.time() - cached_at) < self.cache_ttl_seconds
            is_same_version = (current_version is None or current_version == cached_version)
            if is_fresh and is_same_version:
                return model, preprocessor, metadata
            logger.info(
                "Cache invalidated for '%s' (fresh=%s, same_version=%s, cached_v=%s, current_v=%s)",
                model_name, is_fresh, is_same_version, cached_version, current_version,
            )

        versions = self.registry.list_versions(model_name, stage="production")
        if not versions:
            raise ForecastServiceError(
                f"No production model for '{model_name}'. It may not have "
                "cleared the promotion quality gate yet, or was never trained."
            )
        metadata = versions[0]
        try:
            model = self.registry.load(model_name=model_name, version=metadata["version"])
            preprocessor = self.registry.load_preprocessor(model_name=model_name, version=metadata["version"])
        except Exception as e:
            logger.exception("Failed to load artifact for '%s' v%s", model_name, metadata["version"])
            raise ForecastServiceError(str(e)) from e

        self._model_cache[model_name] = (model, preprocessor, metadata, metadata["version"], time.time())
        return model, preprocessor, metadata

    def run(
        self,
        product_id: str,
        periods: Optional[int] = None,
        raw_future_regressors_df: Optional[pd.DataFrame] = None,  # RENAMED: raw, uncleaned input
    ) -> ForecastServiceResult:
        model_name = self._determine_model_name(product_id)
        model, preprocessor, metadata = self._load_production_model(model_name)
        resolved_periods = self._resolve_periods(periods)

        future_regressors_df = self._clean_future_regressors(
            preprocessor, raw_future_regressors_df, metadata
        )

        future_df = self._generate_future_dataframe(model, resolved_periods, future_regressors_df, metadata)
        raw_prediction = self._run_prediction(model, future_df, product_id)
        future_only = self._extract_future_forecast(raw_prediction, resolved_periods)
        self._validate_output(future_only, resolved_periods, product_id)

        return ForecastServiceResult(
            product_id=product_id, model_name=model_name, model_version=str(metadata["version"]),
            periods=resolved_periods, forecast=future_only, generated_at=pd.Timestamp.utcnow().isoformat(),
        )

        # ---------- Batch forecasting across all products ----------

    def run_batch_forecast(
        self,
        periods: Optional[int] = None,
        raw_future_regressors_by_product: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Runs a forecast for every product currently at stage='production'.
        Always returns a dict, never None or a partial crash -- one
        product's failure (missing model, bad regressor data, Prophet
        error) is recorded and skipped, never discards the others already
        completed. Mirrors train_all_products' per-product try/except
        isolation pattern.
        """
        product_ids = self.registry.list_product_models(
            stage="production", prefix=self.model_name_prefix
        )
        raw_future_regressors_by_product = raw_future_regressors_by_product or {}
        results: Dict[str, Dict[str, Any]] = {}

        logger.info("Starting batch forecast for %d product(s).", len(product_ids))

        for product_id in product_ids:
            try:
                result = self.run(
                    product_id=product_id,
                    periods=periods,
                    raw_future_regressors_df=raw_future_regressors_by_product.get(product_id),
                )
                results[product_id] = {"status": "success", "result": result}
            except ForecastServiceError as e:
                logger.error("Batch forecast failed for '%s': %s", product_id, e)
                results[product_id] = {"status": "failed", "error": str(e)}
            except Exception as e:
                logger.exception("Unexpected error forecasting '%s'", product_id)
                results[product_id] = {"status": "failed", "error": f"Unexpected: {e}"}

        succeeded = sum(1 for r in results.values() if r["status"] == "success")
        logger.info(
            "Batch forecast complete — %d/%d products succeeded.",
            succeeded, len(product_ids),
        )

        return results

    def _clean_future_regressors(
        self, preprocessor, raw_df: Optional[pd.DataFrame], metadata: Dict[str, Any],
    ) -> Optional[pd.DataFrame]:
       
        if raw_df is None:
            return None
        if preprocessor is None:
            raise ForecastServiceError(
                f"Model '{metadata['model_name']}' v{metadata['version']} requires "
                "regressor cleaning but no preprocessor artifact was found for it. "
                "It may have been registered before preprocessor persistence was added."
            )

        ds_col = raw_df["ds"] if "ds" in raw_df.columns else None
        pl_df = pl.from_pandas(raw_df.drop(columns=["ds"], errors="ignore"))

        try:
            cleaned_pl, _ = preprocessor.transform(pl_df)
        except Exception as e:
            logger.exception("Preprocessor.transform() failed on incoming regressor data")
            raise ForecastServiceError(f"Failed to clean incoming regressor data: {e}") from e

        cleaned_pd = cleaned_pl.to_pandas()
        if ds_col is not None:
            cleaned_pd.insert(0, "ds", ds_col.values)
        return cleaned_pd

    
    def _determine_model_name(self, product_id: str) -> str:
        if not product_id or not isinstance(product_id, str):
            raise ForecastServiceError(f"Invalid product_id: {product_id!r}")
        return f"{self.model_name_prefix}{product_id}"

    

    def _resolve_periods(self, periods: Optional[int]) -> int:
        resolved = periods if periods is not None else self.default_forecast_horizon
        if resolved <= 0:
            raise ForecastServiceError(f"periods must be positive, got {resolved}.")
        return resolved

    def _generate_future_dataframe(
        self, model, periods: int, future_regressors_df: Optional[pd.DataFrame],
        metadata: Dict[str, Any],
    ) -> pd.DataFrame:
        parameters = metadata.get("parameters") or {}
        expected_regressors = parameters.get("extra_regressors") or []
        fitted_freq = getattr(model, "_fitted_freq", None) or self.default_forecast_freq

        future_df = model.make_future_dataframe(
            periods=periods, freq=fitted_freq, include_history=False,
        )

        if expected_regressors:
            if future_regressors_df is None:
                raise ForecastServiceError(
                    f"Model requires regressors {expected_regressors} for future periods."
                )
            missing_cols = set(expected_regressors) - set(future_regressors_df.columns)
            if missing_cols:
                raise ForecastServiceError(f"future_regressors_df missing column(s): {sorted(missing_cols)}")
            if "ds" not in future_regressors_df.columns:
                raise ForecastServiceError("future_regressors_df must contain a 'ds' column.")

            missing_dates = sorted(set(future_df["ds"]) - set(future_regressors_df["ds"]))
            if missing_dates:
                raise ForecastServiceError(
                    f"Missing {len(missing_dates)} required future date(s): "
                    f"{missing_dates[:5]}{'...' if len(missing_dates) > 5 else ''}"
                )

            future_df = future_df.merge(future_regressors_df, on="ds", how="left")
            null_check = future_df[expected_regressors].isna().any()
            bad_cols = null_check[null_check].index.tolist()
            if bad_cols:
                raise ForecastServiceError(
                    f"Null regressor values after merge in {bad_cols} "
                    "(check for duplicate 'ds' rows in future_regressors_df)."
                )
        return future_df

    def _run_prediction(self, model, future_df: pd.DataFrame, product_id: str):
        try:
            return model.predict(future_df)
        except Exception:
            logger.exception("predict() failed for product_id=%s", product_id)
            raise ForecastServiceError(f"Prediction failed for {product_id}")

    def _extract_future_forecast(self, prediction: pd.DataFrame, periods: int) -> pd.DataFrame:
        return prediction.tail(periods).reset_index(drop=True)

    def _validate_output(self, forecast: pd.DataFrame, periods: int, product_id: str) -> None:
        missing = {"ds", "yhat"} - set(forecast.columns)
        if missing:
            raise ForecastServiceError(f"Missing column(s) {missing} for {product_id}")
        if len(forecast) != periods:
            raise ForecastServiceError(f"Got {len(forecast)} forecast rows, expected {periods}, for {product_id}")
        if forecast["yhat"].isna().any():
            raise ForecastServiceError(f"Null yhat values in forecast for {product_id}")
        if forecast["ds"].duplicated().any():
            raise ForecastServiceError(f"Duplicate 'ds' values in forecast for {product_id}")