from __future__ import annotations

import io
import polars as pl
import logging
import pickle
import uuid
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

        try:
            PostgresClient.execute("ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS training_data_key TEXT;")
            logger.info("Successfully migrated model_registry schema with tracking column.")
        except Exception as migration_error:
            logger.warning(f"Migration check completed or skipped: {migration_error}")
            
    
    def _sanitize(self, obj: Any) -> Any:
        """
        Recursively replace NaN or Infinity with None for JSON compliance.
        """
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._sanitize(x) for x in obj]
        return obj
    

    def register(
        self,
        model: Any,
        model_name: str,
        metrics: Dict[str, float],
        parameters: Dict[str, Any],
        problem_type: str,
        training_data_key: str | None = None,
        stage: str = "staging",
    ) -> Dict[str, Any]:
        """
        Serialize *model*, upload it to S3, and record metadata in Postgres.

        Returns
        The metadata dict for the new version (mirrors the DB row).
        """
        clean_metrics = self._sanitize(metrics)
        clean_parameters = self._sanitize(parameters)

        version = self._next_version(model_name)
        artifact_key = self._artifact_key(model_name, version)
        run_id = str(uuid.uuid4())
        
        model_bytes = pickle.dumps(model)
        self._s3._client.put_object(
            Bucket=self._s3.bucket_name,
            Key=artifact_key,
            Body=model_bytes,
            ContentType="application/octet-stream",
            Metadata={
                "model_name": model_name,
                "version": str(version),
                "run_id": run_id,
            },
        )
        logger.info(
            "Model artifact uploaded",
            extra={"bucket": self._s3.bucket_name, "key": artifact_key},
        )

        #  Persist metadata to Postgres 
        row = PostgresClient.execute(
            """
            INSERT INTO model_registry
                (run_id, model_name, version, stage, problem_type,
                 metrics, parameters, artifact_key, training_data_key, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                run_id,
                model_name,
                version,
                stage,
                problem_type,
                json.dumps(clean_metrics),
                json.dumps(clean_parameters),
                artifact_key,
                training_data_key,
                datetime.utcnow(),
            ),
            returning=True,
        )

        logger.info(
            "Model registered",
            extra={"model_name": model_name, "version": version, "stage": stage},
        )
        return row  # type: ignore[return-value]

    def load(
        self,
        model_name: str,
        version: Optional[int] = None,
        stage: Optional[str] = None,
    ) -> Any:
        """
        Download a model artifact from S3 and deserialize it.

        Resolution order (most-specific wins):
        1. *model_name* + *version*  → exact match
        2. *model_name* + *stage*    → latest version in that stage
        3. *model_name* only         → latest version overall
        """
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
        """
        Load the training sample associated with the
        production model.
        """

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

    def promote(
        self,
        model_name: str,
        version: int,
        new_stage: str,
    ) -> None:
        """
        Update the *stage* of a specific model version in Postgres.

        Raises ``ValueError`` if the (model_name, version) pair does not exist.
        """
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
        """
        Return all registered versions of *model_name*, newest first.

        Optionally filter by *stage*.
        """
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
        """Create the ``model_registry`` table if it does not already exist."""
        PostgresClient.execute(_CREATE_TABLE_SQL)
        logger.debug("model_registry table ensured")

    def _artifact_key(self, model_name: str, version: int) -> str:
        return f"{self._s3_prefix}/{model_name}/v{version}/model.pkl"
    
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
        """
        Look up a single metadata row from Postgres.

        Resolution order:
        1. Exact (model_name, version)
        2. Latest version in *stage*
        3. Overall latest version
        """
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