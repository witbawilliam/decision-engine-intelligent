"""
dependencies.py

Central dependency providers for the ML platform.
"""

from functools import lru_cache

from config import get_settings
from storage.s3_client import S3Client
from storage.redis_client import RedisClient
from storage.postgres_client import PostgresClient


@lru_cache
def get_s3_client() -> S3Client:
    """
    Returns shared S3 client.
    """
    settings = get_settings()

    return S3Client(
        endpoint_url=settings.s3.endpoint_url,
        access_key=settings.s3.access_key,
        secret_key=settings.s3.secret_key,
        bucket_name=settings.s3.bucket_name,
    )


@lru_cache
def get_redis_client() -> RedisClient:
    """
    Returns Redis client instance.
    """

    settings = get_settings()

    return RedisClient(
        host=settings.redis.host,
        port=settings.redis.port,
        db=settings.redis.db,
        password=settings.redis.password,
    )


@lru_cache
def get_postgres_client() -> PostgresClient:
    """
    Returns PostgreSQL client.
    """

    settings = get_settings()

    return PostgresClient(
        host=settings.postgres.host,
        port=settings.postgres.port,
        user=settings.postgres.user,
        password=settings.postgres.password,
        database=settings.postgres.database,
    )