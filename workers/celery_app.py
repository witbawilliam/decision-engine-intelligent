from __future__ import annotations

import logging
from typing import Any

from celery import Celery
from celery.result import AsyncResult
from celery.signals import worker_init, worker_shutdown

from workers.celery_config import CelerySettings

logger = logging.getLogger(__name__)

def _create_app() -> Celery:
    """
    Build and configure the Celery application.
    """
    app = Celery("automl_platform")
    app.config_from_object(CelerySettings)

    task_modules = ["tasks_validation", "tasks_training"]
    
    for module in task_modules:
        app.autodiscover_tasks(
            packages=["workers"],
            related_name=module,
            force=True,
        )

    return app

celery_app = _create_app()



def dispatch_automl_task(task_type: str, payload: dict[str, Any]) -> AsyncResult:
    """
    Dispatches tasks by string name to avoid imports in routes_jobs.py.
    
    The keys here map to the 'name' argument in your @celery_app.task decorators.
    """
    task_mapping = {
        "validate": "workers.tasks_validation.task_validation",
        "train": "workers.tasks_training.tabular_task",
        "forecast":  "workers.tasks_training.train_all_products_task"
        
    }
    
    task_name = task_mapping.get(task_type)
    if not task_name:
        raise ValueError(f"Unknown task type: {task_type}. Choose from: {list(task_mapping.keys())}")
        
    # Dispatches the task to the broker (Redis/RabbitMQ)
    return celery_app.send_task(task_name, args=[payload])

# Attach helper to the app object
celery_app.dispatch_automl_task = dispatch_automl_task



@worker_init.connect
def _on_worker_init(sender: object, **kwargs: object) -> None:
    logger.info(
        "Celery worker initialised",
        extra={
            "broker": CelerySettings.broker_url,
            "queues": [q.name for q in CelerySettings.task_queues],
            "concurrency": CelerySettings.worker_concurrency,
        },
    )

@worker_shutdown.connect
def _on_worker_shutdown(sender: object, **kwargs: object) -> None:
    logger.info("Celery worker shutting down — flushing telemetry.")

if __name__ == "__main__":
    celery_app.start()