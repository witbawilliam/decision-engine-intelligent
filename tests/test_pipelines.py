from __future__ import annotations

import logging
import pickle
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from storage.postgres_client import PostgresClient
from storage.s3_client import S3Client

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
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (model_name, version)
);
"""


class ModelRegistry:
    """
    ML model registry backed by S3 (artifacts) and PostgreSQL (metadata).

    Artifact storage
    ----------------
    Model pickles are written to S3 under the key pattern::

        {s3_prefix}/{model_name}/v{version}/model.pkl

    Metadata storage
    ----------------
    Every registered version is a row in the ``model_registry`` table.
    The table is created automatically on first instantiation (idempotent).

    Parameters
    ----------
    s3_client:
        A configured :class:`S3Client` instance.  Required for ``register``
        and ``load``; omit only when the instance will be fully mocked in
        tests or when only metadata queries are needed.
    s3_prefix:
        Key prefix inside the bucket (default: ``"ml_registry"``).
    base_path:
        Legacy parameter — previously the registry stored artifacts on local
        disk under this path.  Accepted so existing call-sites such as
        ``ModelRegistry(base_path="ml_registry")`` do not raise
        ``TypeError``.  When *s3_client* is provided, *base_path* is used
        as the fallback *s3_prefix* if no explicit prefix was given.
    """

    def __init__(
        self,
        s3_client: Optional[S3Client] = None,
        s3_prefix: str = "ml_registry",
        base_path: str = "ml_registry",
    ) -> None:
        self._s3: Optional[S3Client] = s3_client
        # Honour base_path as the S3 prefix when no explicit s3_prefix given.
        self._s3_prefix = (s3_prefix if s3_prefix != "ml_registry" else base_path).rstrip("/")
        self._ensure_table()

    
    def register(
        self,
        model: Any,
        model_name: str,
        metrics: Dict[str, float],
        parameters: Dict[str, Any],
        problem_type: str,
        stage: str = "staging",
    ) -> Dict[str, Any]:
        """
        Serialize *model*, upload it to S3, and record metadata in Postgres.

        Returns
        -------
        The metadata dict for the new version (mirrors the DB row).
        """
        import json

        s3 = self._require_s3()
        version = self._next_version(model_name)
        artifact_key = self._artifact_key(model_name, version)
        run_id = str(uuid.uuid4())

        # --- Upload artifact to S3 ---
        model_bytes = pickle.dumps(model)
        s3._client.put_object(
            Bucket=s3.bucket_name,
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
            extra={"bucket": s3.bucket_name, "key": artifact_key},
        )

        # --- Persist metadata to Postgres ---
        row = PostgresClient.execute(
            """
            INSERT INTO model_registry
                (run_id, model_name, version, stage, problem_type,
                 metrics, parameters, artifact_key, created_at)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s)
            RETURNING *
            """,
            (
                run_id,
                model_name,
                version,
                stage,
                problem_type,
                json.dumps(metrics),
                json.dumps(parameters),
                artifact_key,
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
        s3 = self._require_s3()
        meta = self._resolve_metadata(model_name, version=version, stage=stage)
        artifact_key: str = meta["artifact_key"]

        response = s3._client.get_object(
            Bucket=s3.bucket_name,
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
        """Return all registered versions of *model_name*, newest first."""
        if stage:
            return PostgresClient.query(
                "SELECT * FROM model_registry "
                "WHERE model_name = %s AND stage = %s ORDER BY version DESC",
                (model_name, stage),
            )
        return PostgresClient.query(
            "SELECT * FROM model_registry "
            "WHERE model_name = %s ORDER BY version DESC",
            (model_name,),
        )

    def get_metadata(self, model_name: str, version: int) -> Dict[str, Any]:
        """Return the metadata row for a specific (model_name, version) pair."""
        return self._resolve_metadata(model_name, version=version)

    def ping(self) -> Dict[str, bool]:
        """Health-check both backing services."""
        return {
            "postgres": PostgresClient.ping(),
            "s3": self._s3.ping() if self._s3 is not None else False,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_s3(self) -> S3Client:
        """Return the S3 client, or raise a clear error if none was supplied."""
        if self._s3 is None:
            raise RuntimeError(
                "This ModelRegistry has no S3Client. "
                "Pass s3_client=<S3Client> to the constructor."
            )
        return self._s3

    def _ensure_table(self) -> None:
        """Create the ``model_registry`` table if it does not already exist."""
        PostgresClient.execute(_CREATE_TABLE_SQL)
        logger.debug("model_registry table ensured")

    def _artifact_key(self, model_name: str, version: int) -> str:
        return f"{self._s3_prefix}/{model_name}/v{version}/model.pkl"

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