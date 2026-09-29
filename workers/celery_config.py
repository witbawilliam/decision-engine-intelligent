
from __future__ import annotations

import json
import os
from kombu import Exchange, Queue



def _env_int(key: str, default: int) -> int:
    """Read an integer from the environment, falling back to *default*."""
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {key!r} must be an integer, got {raw!r}") from exc


def _env_bool(key: str, default: bool) -> bool:
    """Read a boolean from the environment (accepts 'true'/'false', case-insensitive)."""
    raw = os.getenv(key)
    if raw is None:
        return default
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    raise ValueError(f"Environment variable {key!r} must be 'true' or 'false', got {raw!r}")


def _env_json(key: str, default: dict) -> dict:
    """Read a JSON-encoded dict from the environment."""
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Environment variable {key!r} contains invalid JSON: {exc}") from exc



# Direct exchanges give full control over queue routing without topic overhead.

_default_exchange  = Exchange("default",  type="direct")
_training_exchange = Exchange("training", type="direct")
_validation_exchange = Exchange("validation", type="direct")
_inference_exchange = Exchange("inference", type="direct")




class CelerySettings:
    """
    Celery configuration for ML workloads.

    All class attributes map directly to Celery's lowercase config keys
    (Celery 4+ style).  ``celery_app.config_from_object(CelerySettings)``
    picks them up automatically.
    """


    # Redis is the default for ML workloads — fast, supports pub/sub for
    # task-state streaming, and requires no schema management.
    broker_url: str = os.getenv("CELERY__BROKER_URL", "redis://localhost:6379/0")
    result_backend: str = os.getenv("CELERY__RESULT_BACKEND", "redis://localhost:6379/1")
    # Use a separate Redis DB for results so a FLUSHDB on the broker
    # doesn't wipe pending task results (and vice-versa).

    # Heartbeat interval (seconds) for broker connection health checks.
    # Lower values detect dead connections faster at the cost of more traffic.
    broker_heartbeat: int = 10

    # How long (seconds) to wait for the broker connection on startup before
    # raising. Prevents silent hangs in containerised environments.
    broker_connection_timeout: float = 4.0

    # Retry the broker connection on startup (useful when Redis/RabbitMQ
    # starts slightly after the Celery worker in docker-compose).
    broker_connection_retry_on_startup: bool = True

    # Transport-level options (e.g., socket_timeout, visibility_timeout for SQS).
    # Override via CELERY_BROKER_TRANSPORT_OPTIONS='{"socket_timeout": 5}'.
    broker_transport_options: dict = _env_json(
        "CELERY_BROKER_TRANSPORT_OPTIONS",
        default={
            # Visibility timeout must be >= the longest task time_limit.
            # 8 hours covers forecasting jobs (7200 s) with headroom.
            "visibility_timeout": 28_800,
        },
    )

    # How long (seconds) task results are retained in the backend.
    # 24 hours is enough for async callers to poll; tune down if storage is costly.
    result_expires: int = _env_int("CELERY_RESULT_EXPIRES", default=86_400)

    

    task_queues: tuple = (
        Queue(
            "default",
            exchange=_default_exchange,
            routing_key="default",
        ),
        Queue(
            "validation",
            exchange=_validation_exchange,
            routing_key="validation",
        ),
        Queue(
            "training",
            exchange=_training_exchange,
            routing_key="training",
        ),
        Queue("inference",   exchange=_inference_exchange,   routing_key="inference"),
    )

    
    task_default_queue: str = "default"
    task_default_exchange: str = "default"
    task_default_routing_key: str = "default"

    
    
    task_routes: dict = {
        "workers.tasks_validation.task_validation": {
            "queue": "validation",
            "routing_key": "validation",
        },
        "workers.tasks_training.tabular_task": {
            "queue": "training",
            "routing_key": "training",
        },

        "workers.tasks_training.temporal_task":{
            "queue": "training",
            "routing_key": "training",
        },

        "workers.tasks_forecast_inference.run_batch_forecast_task": {
                "queue": "inference", "routing_key": "inference",
        },
       
    }

    
    worker_max_tasks_per_child: int = _env_int("WORKER_MAX_TASKS_PER_CHILD", default=10)

    
    worker_max_memory_per_child: int | None = (
        _env_int("WORKER_MAX_MEMORY_PER_CHILD", default=0) or None
    )

    worker_pool: str = os.getenv("WORKER_POOL", "prefork")

    
    worker_concurrency: int | None = (
        _env_int("WORKER_CONCURRENCY", default=0) or None
        # None → Celery uses os.cpu_count()
    )

    
    worker_prefetch_multiplier: int = 1

    
    # Set to the longest expected task (forecasting: 2 h) as a safety net.
    task_time_limit: int = _env_int("TASK_TIME_LIMIT", default=7200)

    
    task_soft_time_limit: int = _env_int("TASK_SOFT_TIME_LIMIT", default=7140)

    

    
    # If the worker crashes mid-training, the task is requeued automatically.
    task_acks_late: bool = True

    # Re-queue the task if the worker process is killed (OOM killer, SIGKILL).
    # Works together with task_acks_late.
    task_reject_on_worker_lost: bool = True

    # Store task exceptions in the result backend so callers can inspect them.
    task_store_errors_even_if_ignored: bool = True

    

   
    task_serializer: str = "json"
    result_serializer: str = "json"
    accept_content: list[str] = ["json"]

    

   
    task_track_started: bool = True

    # Send task events to Celery Flower / monitoring tools.
    worker_send_task_events: bool = True
    task_send_sent_event: bool = True

    task_always_eager: bool = _env_bool("TASK_ALWAYS_EAGER", default=False)

    task_eager_propagates: bool = True