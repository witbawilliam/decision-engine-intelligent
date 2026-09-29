from __future__ import annotations

import io
import polars as pl
import logging
import pickle
import uuid
import re  # Added for production string parsing
from datetime import datetime
from typing import Any, Dict, Optional

from storage.postgres_client import PostgresClient
from storage.s3_client import S3Client
import math
import json

logger = logging.getLogger(__name__)


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS model_registry (
    id            SERIAL       PRIMARY KEY,
    run_id        UUID         NOT NULL UNIQUE,
    model_name    TEXT         NOT NULL,
    version       INTEGER      NOT NULL,
    stage         TEXT         NOT NULL DEFAULT 'staging',
    problem_type  TEXT         NOT NULL,
    metrics       JSONB        NOT NULL DEFAULT '{}',
    parameters    JSONB        NOT NULL DEFAULT '{}',
    artifact_key  TEXT         NOT NULL,
    training_data_key TEXT,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (model_name, version)
);
"""

_CREATE_PREDICTION_AUDIT_SQL = """
CREATE TABLE IF NOT EXISTS prediction_audit (
    id SERIAL PRIMARY KEY,

    request_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT,

    features JSONB,

    prediction TEXT,
    risk_score DOUBLE PRECISION,

    latency_ms DOUBLE PRECISION,

    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_CREATE_PREDICTION_FILES_SQL = """
CREATE TABLE IF NOT EXISTS prediction_files (
    id SERIAL PRIMARY KEY,
    job_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT,

    prediction_file_key TEXT NOT NULL,

    row_count INTEGER,

    created_at TIMESTAMPTZ DEFAULT NOW()
);
"""

_CREATE_MODEL_EVALUATIONS_SQL = """
CREATE TABLE IF NOT EXISTS model_evaluations (
    id             SERIAL PRIMARY KEY,
    model_name     TEXT NOT NULL,
    model_version  TEXT NOT NULL,
    problem_type   TEXT,
    predicted      DOUBLE PRECISION NOT NULL,
    actual         DOUBLE PRECISION,
    created_at     TIMESTAMPTZ DEFAULT NOW()
    
);
CREATE INDEX IF NOT EXISTS idx_me_model_name ON model_evaluations (model_name);
CREATE INDEX IF NOT EXISTS idx_me_created_at ON model_evaluations (created_at DESC);
"""

_CREATE_FORECAST_EVALUATIONS_SQL = """
CREATE TABLE IF NOT EXISTS forecast_evaluations (
    id               SERIAL PRIMARY KEY,
    model_name       TEXT NOT NULL,
    model_version    TEXT NOT NULL,
    problem_type     TEXT DEFAULT 'FORECASTING',
    predicted        DOUBLE PRECISION NOT NULL,
    actual           DOUBLE PRECISION,
    prediction_date  TEXT,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_fe_model_name ON forecast_evaluations (model_name);
CREATE INDEX IF NOT EXISTS idx_fe_date ON forecast_evaluations (prediction_date);
"""

_CREATE_FORECAST_PREDICTIONS_SQL = """
CREATE TABLE IF NOT EXISTS forecast_predictions (
    id              SERIAL PRIMARY KEY,
    job_id          TEXT NOT NULL,
    product_id      TEXT NOT NULL,
    model_name      TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    prediction_date TEXT NOT NULL,
    yhat            DOUBLE PRECISION NOT NULL,
    yhat_lower      DOUBLE PRECISION,
    yhat_upper      DOUBLE PRECISION,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_fp_job_id ON forecast_predictions (job_id);
CREATE INDEX IF NOT EXISTS idx_fp_model_name ON forecast_predictions (model_name);
CREATE INDEX IF NOT EXISTS idx_fp_product_id ON forecast_predictions (product_id);
CREATE INDEX IF NOT EXISTS idx_fp_prediction_date ON forecast_predictions (prediction_date);
"""


class ModelRegistry:
   
    def __init__(
        self,
        s3_client: Optional[S3Client] = None,
        s3_prefix: str = "ml_registry",
        base_path: str = "ml_registry",
    ) -> None:
        if s3_client is None:
            raise ValueError(
                "s3_client is required. "
                "Pass an S3Client instance: ModelRegistry(s3_client=my_s3_client)"
            )
        self._s3 = s3_client
        self._s3_prefix = s3_prefix.rstrip("/")
        self._ensure_table()

       
            
    def _sanitize(self, obj: Any) -> Any:
        """Recursively replace NaN or Infinity with None for JSON compliance."""
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._sanitize(x) for x in obj]
        return obj

    def _parse_registration_path(self, model_name: str, training_data_key: str | None) -> tuple[str, str | None]:
        if "datasets/" in model_name or "prophet_datasets/" in model_name:
            match = re.search(r"^(.*\.parquet)", model_name)
            extracted_key = match.group(1) if match else model_name.split(".parquet")[0] + ".parquet"

            suffix = model_name.split(".parquet_")[-1] if ".parquet_" in model_name else "model"

            file_base_name = extracted_key.split("/")[-1].replace(".parquet", "")
            
            if len(file_base_name) > 25:  
                clean_name = f"auto_{suffix}_model"
            else:
                clean_name = f"{file_base_name}_{suffix}"

            if not training_data_key:
                training_data_key = extracted_key

            return clean_name, training_data_key

        return model_name, training_data_key

    def register(
        self,
        model: Any,
        model_name: str,
        metrics: Dict[str, float],
        parameters: Dict[str, Any],
        problem_type: str,
        training_data_key: str | None = None,
        stage: str = "staging",
        preprocessor: Any = None,   # NEW
    ) -> Dict[str, Any]:
        """
        Serialize *model*, upload it to S3, and record metadata in Postgres.
        If *preprocessor* is provided (a fitted FeatureProcessor), it's pickled
        and uploaded alongside the model so inference can reproduce identical
        imputation/encoding/scaling on new data -- without this, transform()
        at inference time would either be impossible or silently use different
        fill values / encoding maps than what the model was trained on.
        """
        model_name, training_data_key = self._parse_registration_path(model_name, training_data_key)
        clean_metrics = self._sanitize(metrics)
        clean_parameters = self._sanitize(parameters)

        version = self._next_version(model_name)
        artifact_key = self._artifact_key(model_name, version)
        run_id = str(uuid.uuid4())

        model_bytes = pickle.dumps(model)
        self._s3._client.put_object(
            Bucket=self._s3.bucket_name, Key=artifact_key, Body=model_bytes,
            ContentType="application/octet-stream",
            Metadata={"model_name": model_name, "version": str(version), "run_id": run_id},
        )
        logger.info("Model artifact uploaded", extra={"bucket": self._s3.bucket_name, "key": artifact_key})

        preprocessor_key = None
        if preprocessor is not None:
            preprocessor_key = self._preprocessor_key(model_name, version)
            try:
                preprocessor_bytes = pickle.dumps(preprocessor)
                self._s3._client.put_object(
                    Bucket=self._s3.bucket_name, Key=preprocessor_key, Body=preprocessor_bytes,
                    ContentType="application/octet-stream",
                    Metadata={"model_name": model_name, "version": str(version), "run_id": run_id},
                )
                logger.info(
                    "Preprocessor artifact uploaded",
                    extra={"bucket": self._s3.bucket_name, "key": preprocessor_key},
                )
            except Exception:
                logger.exception(
                    "Failed to upload preprocessor for '%s' v%s -- inference will not "
                    "be able to reproduce training-time cleaning for this version.",
                    model_name, version,
                )
                preprocessor_key = None  # don't record a key that was never actually written

        row = PostgresClient.execute(
            """
            INSERT INTO model_registry
                (run_id, model_name, version, stage, problem_type,
                metrics, parameters, artifact_key, training_data_key, preprocessor_key, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                run_id, model_name, version, stage, problem_type,
                json.dumps(clean_metrics), json.dumps(clean_parameters),
                artifact_key, training_data_key, preprocessor_key,
                datetime.utcnow(),
            ),
            returning=True,
        )

        logger.info("Model registered", extra={"model_name": model_name, "version": version, "stage": stage})
        return row

    def load(
        self,
        model_name: str,
        version: Optional[int] = None,
        stage: Optional[str] = None,
    ) -> Any:
        """Download a model artifact from S3 and deserialize it."""
        meta = self._resolve_metadata(model_name, version=version, stage=stage)
        artifact_key: str = meta["artifact_key"]

        response = self._s3._client.get_object(
            Bucket=self._s3.bucket_name,
            Key=artifact_key,
        )
        model_bytes = response["Body"].read()
        model = pickle.loads(model_bytes)

        logger.info(
            "Model loaded",
            extra={
                "model_name": model_name,
                "version": meta["version"],
                "stage": meta["stage"],
            },
        )
        return model
    
    def load_training_sample(
        self,
        model_name: str,
    ) -> pl.DataFrame:
        """Load the training sample associated with the production model."""
        metadata = self._resolve_metadata(
            model_name=model_name,
            stage="production",
        )

        training_key = metadata.get("training_data_key")

        if not training_key:
            raise ValueError(
                f"No training sample found for model '{model_name}'"
            )

        response = self._s3._client.get_object(
            Bucket=self._s3.bucket_name,
            Key=training_key,
        )

        data = response["Body"].read()
        return pl.read_parquet(io.BytesIO(data))

    def load_preprocessor(
        self,
        model_name: str,
        version: Optional[int] = None,
        stage: Optional[str] = None,
    ) -> Any:
        """Download and deserialize the fitted FeatureProcessor for a model
        version. Returns None if no preprocessor was registered for it (e.g.
        older versions registered before this feature existed)."""
        meta = self._resolve_metadata(model_name, version=version, stage=stage)
        preprocessor_key = meta.get("preprocessor_key")

        if not preprocessor_key:
            logger.warning(
                "No preprocessor artifact recorded for '%s' v%s -- was it "
                "registered before preprocessor persistence was added?",
                model_name, meta["version"],
            )
            return None

        response = self._s3._client.get_object(Bucket=self._s3.bucket_name, Key=preprocessor_key)
        preprocessor_bytes = response["Body"].read()
        preprocessor = pickle.loads(preprocessor_bytes)

        logger.info(
            "Preprocessor loaded",
            extra={"model_name": model_name, "version": meta["version"], "stage": meta.get("stage")},
        )
        return preprocessor

    def promote(
        self,
        model_name: str,
        version: int,
        new_stage: str,
    ) -> None:
        """Update the *stage* of a specific model version in Postgres."""
        result = PostgresClient.execute(
            """
            UPDATE model_registry
               SET stage = %s
             WHERE model_name = %s AND version = %s
            RETURNING id
            """,
            (new_stage, model_name, version),
            returning=True,
        )
        if result is None:
            raise ValueError(
                f"No entry found for model '{model_name}' version {version}."
            )
        logger.info(
            "Model stage updated",
            extra={"model_name": model_name, "version": version, "new_stage": new_stage},
        )

    def list_versions(
        self,
        model_name: str,
        stage: Optional[str] = None,
    ) -> list[Dict[str, Any]]:
        """Return all registered versions of *model_name*, newest first."""
        if stage:
            rows = PostgresClient.query(
                """
                SELECT * FROM model_registry
                 WHERE model_name = %s AND stage = %s
                 ORDER BY version DESC
                """,
                (model_name, stage),
            )
        else:
            rows = PostgresClient.query(
                """
                SELECT * FROM model_registry
                 WHERE model_name = %s
                 ORDER BY version DESC
                """,
                (model_name,),
            )
        return rows

    def get_metadata(
        self,
        model_name: str,
        version: int,
    ) -> Dict[str, Any]:
        """Return the metadata row for a specific (model_name, version) pair."""
        return self._resolve_metadata(model_name, version=version)

    def ping(self) -> Dict[str, bool]:
        """Health-check both backing services."""
        return {
            "postgres": PostgresClient.ping(),
            "s3": self._s3.ping(),
        }
    
    # Internal helpers

    def _ensure_table(self) -> None:
        PostgresClient.execute(_CREATE_TABLE_SQL)
        PostgresClient.execute(_CREATE_PREDICTION_AUDIT_SQL)
        PostgresClient.execute(_CREATE_PREDICTION_FILES_SQL)
        PostgresClient.execute(_CREATE_MODEL_EVALUATIONS_SQL)
        PostgresClient.execute(_CREATE_FORECAST_EVALUATIONS_SQL)
        PostgresClient.execute(_CREATE_FORECAST_PREDICTIONS_SQL)

        logger.debug("model_registry table ensured")
        logger.debug("prediction_audit table ensured")

        try:

            PostgresClient.execute("ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS training_data_key TEXT;")
            PostgresClient.execute("ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS preprocessor_key TEXT;")
            logger.info("Successfully migrated model_registry schema with tracking column.")
        except Exception as migration_error:
            logger.warning(f"Migration check completed or skipped: {migration_error}")

    def _artifact_key(self, model_name: str, version: int) -> str:
        return f"{self._s3_prefix}/{model_name}/v{version}/model.pkl"

    def _preprocessor_key(self, model_name: str, version: int) -> str:
        return f"{self._s3_prefix}/{model_name}/v{version}/preprocessor.pkl"
    
    def _training_sample_key(
        self,
        model_name: str,
        version: int,
    ) -> str:
        return (
            f"{self._s3_prefix}/"
            f"{model_name}/"
            f"v{version}/"
            f"training_sample.parquet"
        )

    def _next_version(self, model_name: str) -> int:
        """Return the next integer version for *model_name* (1-indexed)."""
        rows = PostgresClient.query(
            "SELECT COALESCE(MAX(version), 0) AS max_version "
            "FROM model_registry WHERE model_name = %s",
            (model_name,),
        )
        return (rows[0]["max_version"] or 0) + 1

    def _resolve_metadata(
        self,
        model_name: str,
        version: Optional[int] = None,
        stage: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Look up a single metadata row from Postgres."""
        if version is not None:
            rows = PostgresClient.query(
                "SELECT * FROM model_registry "
                "WHERE model_name = %s AND version = %s",
                (model_name, version),
            )
        elif stage is not None:
            rows = PostgresClient.query(
                "SELECT * FROM model_registry "
                "WHERE model_name = %s AND stage = %s "
                "ORDER BY version DESC LIMIT 1",
                (model_name, stage),
            )
        else:
            rows = PostgresClient.query(
                "SELECT * FROM model_registry "
                "WHERE model_name = %s "
                "ORDER BY version DESC LIMIT 1",
                (model_name,),
            )

        if not rows:
            raise ValueError(
                f"No model found for name='{model_name}', "
                f"version={version!r}, stage={stage!r}."
            )
        return rows[0]

    def list_product_models(self, stage: str = "production", prefix: str = "prophet_product_") -> list[str]:
        """Return product_ids (prefix stripped) for every model_name currently at `stage`."""
        rows = PostgresClient.query(
            "SELECT DISTINCT model_name FROM model_registry WHERE stage = %s AND model_name LIKE %s",
            (stage, f"{prefix}%"),
        )
        return [r["model_name"].removeprefix(prefix) for r in rows]

    def save_forecast_results(
        self,
        results: Dict[str, Dict[str, Any]],
        job_id: str,
    ) -> int:
    
        rows_written = 0

        for product_id, entry in results.items():
            if entry.get("status") != "success":
                continue

            result = entry["result"]  # ForecastServiceResult
            forecast_df = result.forecast

            for _, row in forecast_df.iterrows():
                try:
                    yhat_lower = row.get("yhat_lower")
                    yhat_upper = row.get("yhat_upper")
                    PostgresClient.execute(
                        """
                        INSERT INTO forecast_predictions
                            (job_id, product_id, model_name, model_version,
                            prediction_date, yhat, yhat_lower, yhat_upper)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            job_id,
                            product_id,
                            result.model_name,
                            result.model_version,
                            str(row["ds"]),
                            float(row["yhat"]),
                            float(yhat_lower) if yhat_lower is not None else None,
                            float(yhat_upper) if yhat_upper is not None else None,
                        ),
                    )
                    rows_written += 1
                except Exception:
                    logger.exception(
                        "Failed to write forecast_predictions row for product_id=%s, "
                        "job_id=%s, date=%s -- continuing with remaining rows.",
                        product_id, job_id, row.get("ds"),
                    )
                    continue  # one bad row shouldn't abort the whole product's write

        logger.info(
            "Saved %d forecast prediction row(s) for job_id=%s across %d product(s).",
            rows_written, job_id, sum(1 for e in results.values() if e.get("status") == "success"),
        )
        return rows_written