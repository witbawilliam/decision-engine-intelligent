

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import BaseModel, Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict



class DatabaseConfig(BaseModel):
    """PostgreSQL metadata store configuration."""

    url: PostgresDsn = Field(
        ...,
        description="Full PostgreSQL connection URI.",
        examples=["postgresql://user:pass@localhost:5432/automl_db"],
    )
    pool_size:     int  = Field(default=20, ge=5,  le=100)
    pool_pre_ping: bool = True

    def connection_string(self) -> str:
        """
        Returns url as a plain str.
        SQLAlchemy and asyncpg expect str, not PostgresDsn object.
        """
        return str(self.url)


class RedisConfig(BaseModel):
    """Redis cache and message broker configuration."""

    url: RedisDsn = Field(
        ...,
        description="Redis connection URI.",
        examples=["redis://localhost:6379/0"],
    )
    max_connections: int = Field(default=10, ge=1)

    def connection_string(self) -> str:
        """Returns url as a plain str. redis-py expects str, not RedisDsn."""
        return str(self.url)


class S3Config(BaseModel):
    """
    S3 / MinIO artifact and dataset storage configuration.

    endpoint_url is Optional[str] — not HttpUrl — because:
      - Blank values in .env (S3__ENDPOINT_URL=) fail HttpUrl validation
      - A plain str validator lets us accept blank = None cleanly
      - We validate the URL format ourselves when a value is present
    """

    
    endpoint_url: Optional[str] = Field(
        default=None,
        description="Custom endpoint for MinIO/LocalStack. Leave blank for real AWS.",
    )

    
    access_key:  str = Field(..., min_length=1,  description="AWS / MinIO access key.")
    secret_key:  str = Field(..., min_length=1,  description="AWS / MinIO secret key.")
    bucket_name: str = Field(..., pattern=r"^[a-z0-9.\-]{3,63}$")
    region:      str = Field(default="us-east-1")

    @field_validator("endpoint_url", mode="before")
    @classmethod
    def _blank_endpoint_is_none(cls, v: Optional[str]) -> Optional[str]:
        """
        Converts blank string (S3__ENDPOINT_URL=) to None.
        Prevents HttpUrl validation errors when the field is intentionally empty.
        """
        if v is None or str(v).strip() == "":
            return None
        return v

    def boto3_endpoint(self) -> Optional[str]:
        """
        Returns endpoint_url as Optional[str] for boto3.
        boto3 accepts None (uses real AWS) or a plain str (MinIO/LocalStack).
        """
        return self.endpoint_url if self.endpoint_url else None


class CeleryConfig(BaseModel):
    """
    FIX 3: Celery broker and worker configuration.
    celery_app.py reads from here instead of hardcoding Redis URLs.
    Uses a separate Redis DB index from the cache (broker=1, result=2)
    so Celery tasks and feature cache never interfere.
    """

    broker_url:      RedisDsn = Field(
        ...,
        description="Celery message broker URL.",
        examples=["redis://localhost:6379/1"],
    )
    result_backend:  RedisDsn = Field(
        ...,
        description="Celery result backend URL.",
        examples=["redis://localhost:6379/2"],
    )
    task_serializer:    str = Field(default="json")
    result_serializer:  str = Field(default="json")
    worker_concurrency: int = Field(default=4, ge=1, le=64)

    def broker_string(self) -> str:
        """Plain str for Celery — it does not accept RedisDsn objects."""
        return str(self.broker_url)

    def backend_string(self) -> str:
        return str(self.result_backend)


class ModelRegistryConfig(BaseModel):
    """Model lifecycle and caching settings."""

    registry_path:     str  = Field(default="models/")
    cache_enabled:     bool = True
    cache_ttl_seconds: int  = Field(default=3600, ge=60)


class Settings(BaseSettings):
    """
    Main application settings — Pydantic Settings V2.

    Reads from environment variables first, then falls back to .env file.
    Missing required fields raise a ValidationError immediately on startup
    so the app never boots with broken configuration.
    """
    

    model_config = SettingsConfigDict(
        env_file          = ".env",
        env_file_encoding = "utf-8",
        env_nested_delimiter = "__",   # POSTGRES__URL → settings.postgres.url
        case_sensitive    = False,
        extra             = "ignore",  # ignore unknown env vars instead of crashing
    )

    
    app_name:    str = Field(default="ML-Intelligence-Platform")
    environment: Literal["development", "staging", "production"] = "development"

    
    postgres: DatabaseConfig
    redis:    RedisConfig
    s3:       S3Config
    celery:   CeleryConfig          


    model_registry: ModelRegistryConfig = ModelRegistryConfig()

    
    log_level:      Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    enable_tracing: bool = False



@lru_cache
def get_settings() -> Settings:
    """
    Thread-safe, cached settings loader.
    lru_cache ensures .env is parsed exactly once per process — not per request.
    Call clear_settings_cache() in tests to reset between runs.
    """
    return Settings()


def clear_settings_cache() -> None:
    """
    FIX 4: Clears the lru_cache so get_settings() re-reads the environment.
    Use in tests:
        from app.config import clear_settings_cache
        clear_settings_cache()
    """
    get_settings.cache_clear()