
from __future__ import annotations

import logging

from celery import Celery
from celery.signals import worker_init, worker_shutdown

from workers.celery_config import CelerySettings

logger = logging.getLogger(__name__)



def _create_app() -> Celery:
    """
    Build and configure the Celery application.

    Keeping construction in a function (rather than bare module-level code)
    makes the app easy to recreate in tests with a different config object.
    """
    app = Celery("automl_platform")

    
    app.config_from_object(CelerySettings)

    app.autodiscover_tasks(
        packages=["workers"],
        related_name="tasks",
        force=True,
    )

    return app




celery_app = _create_app()



@worker_init.connect
def _on_worker_init(sender: object, **kwargs: object) -> None:
    """
    Runs once per worker process after it starts.

    Good place to:
    - Warm up DB / cache connections.
    - Initialise telemetry SDKs (Sentry, Datadog APM).
    - Log the active configuration for auditability.
    """
    logger.info(
        "Celery worker initialised",
        extra={
            "broker": CelerySettings.broker_url,
            "queues": [q.name for q in CelerySettings.task_queues],
            "concurrency": CelerySettings.worker_concurrency,
            "max_tasks_per_child": CelerySettings.worker_max_tasks_per_child,
        },
    )


@worker_shutdown.connect
def _on_worker_shutdown(sender: object, **kwargs: object) -> None:
    """
    Runs once per worker process just before it exits.

    Good place to:
    - Flush buffered telemetry / log handlers.
    - Release GPU memory or file handles held at process level.
    """
    logger.info("Celery worker shutting down — flushing telemetry.")



if __name__ == "__main__":
   
    celery_app.start()