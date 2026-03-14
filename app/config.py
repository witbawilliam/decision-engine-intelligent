"""
config.py

Central configuration for the ML platform.
"""

from functools import lru_cache
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseModel):

    host: str
    port: int
    user: str
    password: str
    database: str


class RedisConfig(BaseModel):

    host: str
    port: int
    db: int = 0
    password: str | None = None


class S3Config(BaseModel):

    endpoint_url: str
    access_key: str
    secret_key: str
    bucket_name: str


class ModelRegistryConfig(BaseModel):

    registry_path: str = "models/"
    cache_enabled: bool = True


class Settings(BaseSettings):
    """
    Main application settings.
    """

    environment: str = Field(default="development")

    # Database
    postgres: DatabaseConfig

    # Redis
    redis: RedisConfig

    # Object storage
    s3: S3Config

    # Model registry
    model_registry: ModelRegistryConfig = ModelRegistryConfig()

    # Monitoring
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings loader.
    """
    return Settings()