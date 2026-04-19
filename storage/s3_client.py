
from __future__ import annotations

import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError
from app.config import get_settings


settings = get_settings()

logger = logging.getLogger(__name__)




_MULTIPART_THRESHOLD = 100 * 1024 * 1024   
_MULTIPART_CHUNKSIZE = 50 * 1024 * 1024    

_TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=_MULTIPART_THRESHOLD,
    multipart_chunksize=_MULTIPART_CHUNKSIZE,
    max_concurrency=10,       
    use_threads=True
)

class S3Client:
    """
    Thin, opinionated wrapper around ``boto3.client("s3")``.

    Design decisions
    ----------------
    * One instance per bucket — keeps the API simple and ensures all
      operations target the same bucket without repeating the name.
    * Connection pooling via ``max_pool_connections`` — critical when many
      Celery workers share the same Python process (prefork copies the pool).
    * Multipart uploads for large files — boto3's default threshold is 8 MiB;
      we raise it to 100 MiB to reduce part-count overhead for model artefacts.
    * Structured logging throughout — no f-strings in ``extra=`` fields.
    * ``_ensure_bucket`` is dev/MinIO only; production buckets are Terraform-managed.
    """

    def __init__(
        self,
        bucket_name: str,
        endpoint_url: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: str = "us-east-1",
        max_pool_connections: int = 50,
    ) -> None:
        self.bucket_name = bucket_name or settings.s3.bucket_name
        self._region = region or settings.s3.region

        resolved_endpoint =endpoint_url or settings.s3.boto3_endpoint()
        resolved_key =access_key or settings.s3.access_key
        resolved_secret =secret_key or settings.s3.secret_key

        self._config = Config(
            region_name=self._region,
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "adaptive"},
            max_pool_connections=max_pool_connections,
        )

        self._client = boto3.client(
            "s3",
            endpoint_url=resolved_endpoint,
            aws_access_key_id=resolved_key,
            aws_secret_access_key=resolved_secret,
            config=self._config,
        )

        if settings.environment != "production":
            self._ensure_bucket()

    

    def _ensure_bucket(self) -> None:
        """
        Verify bucket existence or create it in non-production environments
        (local development, MinIO, CI).

        Production buckets are managed by Terraform/CDK — this method is
        intentionally a no-op there to avoid accidental creation.
        """
        try:
            self._client.head_bucket(Bucket=self.bucket_name)
            logger.debug("Bucket exists", extra={"bucket": self.bucket_name})
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code != "404":
                logger.error(
                    "Bucket access check failed",
                    extra={"bucket": self.bucket_name, "error_code": code},
                )
                raise

           
            create_kwargs: dict[str, Any] = {"Bucket": self.bucket_name}
            if self._region != "us-east-1":
                create_kwargs["CreateBucketConfiguration"] = {
                    "LocationConstraint": self._region
                }

            self._client.create_bucket(**create_kwargs)
            logger.info("Bucket created", extra={"bucket": self.bucket_name, "region": self._region})

    @staticmethod
    def _error_code(exc: ClientError) -> str:
        """Extract the AWS error code string from a ClientError."""
        return exc.response.get("Error", {}).get("Code", "")

    
    def upload_file(
        self,
        local_path: str | Path,
        object_name: str,
        metadata: Optional[dict[str, str]] = None,
        content_type: Optional[str] = None,
    ) -> str:
        """
        Upload a local file to S3 with optional metadata and content-type.

        Files > 100 MiB are uploaded using multipart transfers automatically
        via ``TransferConfig``.

        Parameters
        ----------
        local_path:   Path to the local file.
        object_name:  Destination key in the bucket.
        metadata:     Key/value strings stored alongside the object
                      (e.g. model version, job ID). Must be strings only.
        content_type: MIME type (e.g. ``"application/octet-stream"``).
                      Defaults to boto3's auto-detection.

        Returns
        -------
        The ``object_name`` key, allowing callers to chain: ``uri = client.upload_file(...)``.
        """
        extra_args: dict[str, Any] = {}
        if metadata is not None:
            extra_args["Metadata"] = metadata
        if content_type is not None:
            extra_args["ContentType"] = content_type

        try:
            self._client.upload_file(
                Filename=str(local_path),
                Bucket=self.bucket_name,
                Key=object_name,
                ExtraArgs=extra_args or None,
                Config=_TRANSFER_CONFIG,
            )
            logger.info(
                "File uploaded to S3",
                extra={"bucket": self.bucket_name, "key": object_name, "local_path": str(local_path)},
            )
            return object_name
        except ClientError as exc:
            logger.error(
                "S3 upload failed",
                extra={
                    "bucket": self.bucket_name,
                    "key": object_name,
                    "error_code": self._error_code(exc),
                },
            )
            raise

    def download_file(self, object_name: str, local_path: str | Path) -> Path:
        """
        Download an S3 object to a local path.

        Parameters
        ----------
        object_name:  Source key in the bucket.
        local_path:   Destination on the local filesystem.
                      Parent directories are created if they do not exist.

        Returns
        -------
        ``Path`` to the downloaded file.
        """
        destination = Path(local_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._client.download_file(
                Bucket=self.bucket_name,
                Key=object_name,
                Filename=str(destination),
                Config=_TRANSFER_CONFIG,
            )
            logger.info(
                "File downloaded from S3",
                extra={"bucket": self.bucket_name, "key": object_name, "local_path": str(destination)},
            )
            return destination
        except ClientError as exc:
            logger.error(
                "S3 download failed",
                extra={
                    "bucket": self.bucket_name,
                    "key": object_name,
                    "error_code": self._error_code(exc),
                },
            )
            raise

    def delete_object(self, object_name: str) -> None:
        """
        Permanently delete an object from the bucket.

        S3's ``delete_object`` is idempotent — deleting a key that does not
        exist returns HTTP 204 with no error, so this method never raises for
        missing keys.
        """
        try:
            self._client.delete_object(Bucket=self.bucket_name, Key=object_name)
            logger.info(
                "S3 object deleted",
                extra={"bucket": self.bucket_name, "key": object_name},
            )
        except ClientError as exc:
            logger.error(
                "S3 delete failed",
                extra={
                    "bucket": self.bucket_name,
                    "key": object_name,
                    "error_code": self._error_code(exc),
                },
            )
            raise

    def list_objects(self, prefix: str = "") -> list[str]:
        """
        Return all object keys under *prefix* using a paginator.

        The paginator handles S3's 1,000-key-per-page limit automatically.
        Only ``NoSuchBucket`` is silently treated as an empty list; all other
        errors (access denied, network failure) are re-raised so callers are
        not silently misled by an empty result.

        Parameters
        ----------
        prefix: Key prefix to filter results (e.g. ``"models/forecasting/"``).

        Returns
        -------
        Sorted list of matching keys.
        """
        paginator = self._client.get_paginator("list_objects_v2")
        keys: list[str] = []

        try:
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                keys.extend(obj["Key"] for obj in page.get("Contents", []))
            return sorted(keys)
        except ClientError as exc:
            code = self._error_code(exc)
            if code == "NoSuchBucket":
                logger.warning(
                    "Bucket not found during list",
                    extra={"bucket": self.bucket_name, "prefix": prefix},
                )
                return []
            logger.error(
                "S3 list_objects failed",
                extra={"bucket": self.bucket_name, "prefix": prefix, "error_code": code},
            )
            raise

    def get_object_metadata(self, object_name: str) -> dict[str, str]:
        """
        Return the user-defined metadata dict for *object_name*.

        Returns ``{}`` only when the object does not exist (404).
        All other errors (403 access denied, 5xx, network) are re-raised —
        a permission error and a missing key are not the same problem.

        Returns
        -------
        Dict of string metadata values, or ``{}`` if the key does not exist.
        """
        try:
            response = self._client.head_object(Bucket=self.bucket_name, Key=object_name)
            return response.get("Metadata", {})
        except ClientError as exc:
            code = self._error_code(exc)
            if code in ("404", "NoSuchKey"):
                logger.debug(
                    "Object not found — returning empty metadata",
                    extra={"bucket": self.bucket_name, "key": object_name},
                )
                return {}
            logger.error(
                "get_object_metadata failed",
                extra={"bucket": self.bucket_name, "key": object_name, "error_code": code},
            )
            raise

    def generate_presigned_url(
        self,
        object_name: str,
        expiration: int = 3600,
        http_method: str = "GET",
    ) -> str:
        """
        Generate a time-limited presigned URL for direct client access.

        The URL lets a browser or external service download (or upload) the
        object directly from S3, bypassing your API server — useful for
        large model artefacts and dataset exports.

        Parameters
        ----------
        object_name:  Key of the object in the bucket.
        expiration:   URL validity in seconds (default: 1 hour, max: 7 days).
        http_method:  ``"GET"`` for download, ``"PUT"`` for direct client upload.

        Returns
        -------
        Presigned URL string.
        """
        operation = "get_object" if http_method.upper() == "GET" else "put_object"

        try:
            url: str = self._client.generate_presigned_url(
                ClientMethod=operation,
                Params={"Bucket": self.bucket_name, "Key": object_name},
                ExpiresIn=expiration,
            )
            logger.debug(
                "Presigned URL generated",
                extra={
                    "bucket": self.bucket_name,
                    "key": object_name,
                    "expiration_seconds": expiration,
                    "method": http_method,
                },
            )
            return url
        except ClientError as exc:
            logger.error(
                "Failed to generate presigned URL",
                extra={
                    "bucket": self.bucket_name,
                    "key": object_name,
                    "error_code": self._error_code(exc),
                },
            )
            raise

    def object_exists(self, object_name: str) -> bool:
        """
        Return ``True`` if *object_name* exists in the bucket, ``False`` if not.

        Uses ``head_object`` — cheaper than ``get_object`` since no body is
        transferred. Raises for non-404 errors (access denied, etc.).
        """
        try:
            self._client.head_object(Bucket=self.bucket_name, Key=object_name)
            return True
        except ClientError as exc:
            code = self._error_code(exc)
            if code in ("404", "NoSuchKey"):
                return False
            logger.error(
                "object_exists check failed",
                extra={"bucket": self.bucket_name, "key": object_name, "error_code": code},
            )
            raise

    def ping(self) -> bool:
       
        try:
            self._client.list_objects_v2(Bucket=self.bucket_name, MaxKeys=1)
            return True
        except Exception as exc:
            logger.warning(
                "S3 health check failed",
                extra={"bucket": self.bucket_name, "error": str(exc)},
            )
            return False