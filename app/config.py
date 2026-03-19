"""
config.py
Centralized, production-grade configuration for the ML platform.
"""

from functools import lru_cache
from typing import Literal, Optional
from pydantic import BaseModel, Field, PostgresDsn, RedisDsn, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseModel):
    """Configuration for the primary PostgreSQL metadata store."""
    # Use PostgresDsn for automatic validation of connection strings
    url: PostgresDsn = Field(..., description="Full PostgreSQL connection URI")
    pool_size: int = Field(default=20, ge=5, le=100)
    pool_pre_ping: bool = True


class RedisConfig(BaseModel):
    """Configuration for caching and message brokerage."""
    # Use RedisDsn to ensure the connection string is well-formed
    url: RedisDsn = Field(..., description="Redis connection URI (e.g., redis://user:pass@host:port/db)")
    max_connections: int = Field(default=10, ge=1)


class S3Config(BaseModel):
    """Configuration for artifact and dataset storage (AWS S3 or MinIO)."""
    endpoint_url: Optional[HttpUrl] = Field(None, description="Custom endpoint for MinIO/LocalStack")
    access_key: str = Field(..., min_length=16)
    secret_key: str = Field(..., min_length=16)
    bucket_name: str = Field(..., pattern=r"^[a-z0-9.-]{3,63}$")
    region: str = "us-east-1"


class ModelRegistryConfig(BaseModel):
    """Settings for managing the lifecycle of ML models."""
    registry_path: str = "models/"
    cache_enabled: bool = True
    # Added TTL for cached models to ensure fresh deployments
    cache_ttl_seconds: int = 3600 


class Settings(BaseSettings):
    """
    Main application settings using Pydantic Settings V2.
    Loads from environment variables with a fallback to .env.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False
    )

    app_name: str = "ML-Intelligence-Platform"
    environment: Literal["development", "staging", "production"] = "development"

    # Infrastructure components
    postgres: DatabaseConfig
    redis: RedisConfig
    s3: S3Config
    
    # ML specific logic
    model_registry: ModelRegistryConfig = ModelRegistryConfig()

    # Observability
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    enable_tracing: bool = False  # For OpenTelemetry integration


@lru_cache
def get_settings() -> Settings:
    """
    Thread-safe, cached settings loader. 
    Using lru_cache ensures we only parse environment variables once.
    """
    return Settings()