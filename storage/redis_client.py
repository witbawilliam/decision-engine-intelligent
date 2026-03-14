
from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Generator, Optional

import redis
from redis.exceptions import LockError, RedisError

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
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

        try:
            pool = redis.ConnectionPool.from_url(
                redis_url,
                max_connections=20,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=5,
                retry_on_timeout=True,
            )
            client = redis.Redis(connection_pool=pool)
            # Eagerly verify the connection so errors surface at startup,
            # not on the first real operation.
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
        """
        Return the shared Redis client, creating it on first call.

        Double-checked locking prevents two threads from both seeing
        ``_client is None`` and both calling ``_create_client()``.
        """
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
        """
        Deserialise a JSON string back to a Python object.

        If the stored value is not valid JSON (e.g. was written by another
        client that doesn't use JSON), log a warning and return the raw string
        rather than raising — this is the least-surprise behaviour for a cache.
        """
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
        """
        Retrieve and deserialise the value stored at *key*.

        Returns ``None`` if the key does not exist.
        """
        raw: Optional[str] = cls._get_client().get(key)
        if raw is None:
            return None
        return cls._deserialise(raw, key)

    @classmethod
    def delete(cls, *keys: str) -> int:
        """
        Delete one or more keys.

        Returns the number of keys that were actually deleted.
        """
        if not keys:
            return 0
        return cls._get_client().delete(*keys)

    @classmethod
    def exists(cls, key: str) -> bool:
        """Return ``True`` if *key* exists in Redis."""
        return bool(cls._get_client().exists(key))

    @classmethod
    def set_ttl(cls, key: str, ttl: int) -> bool:
        """
        Set or update the TTL on an existing key.

        Returns ``True`` if the timeout was set, ``False`` if the key does
        not exist.
        """
        if ttl <= 0:
            raise ValueError(f"ttl must be a positive integer, got {ttl!r}")
        return cls._get_client().expire(key, ttl)

    @classmethod
    def increment(cls, key: str, amount: int = 1) -> int:
        """
        Atomically increment the integer stored at *key* by *amount*.

        If the key does not exist it is initialised to ``0`` before
        incrementing.

        Returns the new value.
        """
        return cls._get_client().incr(key, amount)

    
    @classmethod
    def push_queue(cls, queue_name: str, payload: Any) -> None:
        """
        Append *payload* (serialised as JSON) to the tail of *queue_name*.

        Queue keys are stored as ``queue:<queue_name>`` to avoid collisions
        with plain string keys.
        """
        cls._get_client().rpush(
            f"{_QUEUE_PREFIX}{queue_name}",
            cls._serialise(payload),
        )

    @classmethod
    def pop_queue(cls, queue_name: str, timeout: int = 0) -> Optional[Any]:
        """
        Remove and return the head item from *queue_name*.
        """
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
        """
        Acquire a Redis-backed distributed lock for *name*.

        Essential for preventing duplicate ML jobs when multiple Celery
        workers race to process the same task (e.g. idempotency-key checks)
        """
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
        """
        Context manager that yields a Redis pipeline for batch operations.

        Batching multiple commands in a single round-trip is significantly
        faster than issuing them one by one — critical for writing many
        metrics or cache entries at once.
        
        """
        pipe = cls._get_client().pipeline(transaction=transaction)
        try:
            yield pipe
        except RedisError:
            pipe.reset()
            raise
        finally:
            pipe.reset()