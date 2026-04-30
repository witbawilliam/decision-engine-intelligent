
from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Generator, Optional

import redis
from redis.exceptions import LockError, RedisError

from app.config import get_settings
settings = get_settings()


logger = logging.getLogger(__name__)



_QUEUE_PREFIX = "queue:"




class RedisClient:
    _client: Optional[redis.Redis] = None
    _lock: threading.Lock = threading.Lock()

    
    @classmethod
    def _create_client(cls) -> redis.Redis:
        """
        Build a ``redis.Redis`` instance backed by a connection pool.
        """
       
       
        redis_url       = str(settings.redis.url)
        max_connections = settings.redis.max_connections

        try:
            pool = redis.ConnectionPool.from_url(
                redis_url,
                max_connections=max_connections,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=5,
                retry_on_timeout=True,
            )
            client = redis.Redis(connection_pool=pool)
            
            client.ping()
            logger.info("Redis connection pool initialised")
            return client

        except RedisError as exc:
            raise RuntimeError(
                "Failed to connect to Redis. "
                "Check REDIS_URL and network connectivity."
            ) from exc

    @classmethod
    def _get_client(cls) -> redis.Redis:
       
        if cls._client is None:
            with cls._lock:
                if cls._client is None:
                    cls._client = cls._create_client()
        return cls._client

    @staticmethod
    def _serialise(value: Any) -> str:
        """Serialise *value* to a JSON string."""
        try:
            return json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Value is not JSON-serialisable: {type(value).__name__}") from exc

    @staticmethod
    def _deserialise(raw: str, key: str) -> Any:
       
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(
                "Non-JSON value found in Redis; returning raw string",
                extra={"key": key},
            )
            return raw

    
    @classmethod
    def ping(cls) -> bool:
        """Return ``True`` if Redis is reachable, ``False`` otherwise."""
        try:
            return cls._get_client().ping()
        except Exception as exc:
            logger.error("Redis ping failed", extra={"error": str(exc)})
            return False

   

    @classmethod
    def set(
        cls,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """
        Store *value* (serialised as JSON) under *key*.

        """
        if ttl is not None and ttl <= 0:
            raise ValueError(f"ttl must be a positive integer, got {ttl!r}")

        payload = cls._serialise(value)

        if ttl is not None:
            cls._get_client().setex(key, ttl, payload)
        else:
            cls._get_client().set(key, payload)

    @classmethod
    def get(cls, key: str) -> Optional[Any]:
        
        raw: Optional[str] = cls._get_client().get(key)
        if raw is None:
            return None
        return cls._deserialise(raw, key)

    @classmethod
    def delete(cls, *keys: str) -> int:
        
        if not keys:
            return 0
        return cls._get_client().delete(*keys)

    @classmethod
    def exists(cls, key: str) -> bool:
        """Return ``True`` if *key* exists in Redis."""
        return bool(cls._get_client().exists(key))

    @classmethod
    def set_ttl(cls, key: str, ttl: int) -> bool:
        
        if ttl <= 0:
            raise ValueError(f"ttl must be a positive integer, got {ttl!r}")
        return cls._get_client().expire(key, ttl)

    @classmethod
    def increment(cls, key: str, amount: int = 1) -> int:
       
        return cls._get_client().incr(key, amount)

    
    @classmethod
    def push_queue(cls, queue_name: str, payload: Any) -> None:
       
        cls._get_client().rpush(
            f"{_QUEUE_PREFIX}{queue_name}",
            cls._serialise(payload),
        )

    @classmethod
    def pop_queue(cls, queue_name: str, timeout: int = 0) -> Optional[Any]:
        
        full_key = f"{_QUEUE_PREFIX}{queue_name}"

        if timeout > 0:
            # BLPOP returns (key, value) or None on timeout.
            result = cls._get_client().blpop(full_key, timeout=timeout)
            raw = result[1] if result else None
        else:
            raw = cls._get_client().lpop(full_key)

        if raw is None:
            return None
        return cls._deserialise(raw, full_key)

    @classmethod
    def queue_length(cls, queue_name: str) -> int:
        """Return the number of items currently in *queue_name*."""
        return cls._get_client().llen(f"{_QUEUE_PREFIX}{queue_name}")

   

    @classmethod
    @contextmanager
    def lock(
        cls,
        name: str,
        timeout: float = 10.0,
        blocking_timeout: Optional[float] = 5.0,
    ) -> Generator[None, None, None]:
       
        acquired_lock = cls._get_client().lock(
            name,
            timeout=timeout,
            blocking_timeout=blocking_timeout,
        )
        if not acquired_lock.acquire(blocking=True):
            raise LockError(f"Could not acquire Redis lock: {name!r}")

        logger.debug("Redis lock acquired", extra={"lock": name})
        try:
            yield
        finally:
            try:
                acquired_lock.release()
                logger.debug("Redis lock released", extra={"lock": name})
            except LockError:
                # Lock expired before release — not an error, just log.
                logger.warning(
                    "Redis lock expired before explicit release",
                    extra={"lock": name},
                )



    @classmethod
    @contextmanager
    def pipeline(cls, transaction: bool = True) -> Generator[redis.client.Pipeline, None, None]:
       
        pipe = cls._get_client().pipeline(transaction=transaction)
        try:
            yield pipe
        except RedisError:
            pipe.reset()
            raise
        finally:
            pipe.reset()