import os
from celery import Celery

# Setup the Celery instance
celery_app = Celery(
    "automl_worker",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    # This ensures we find tasks in the workers/ folder
    include=["workers.tasks_training"]
)

# Optimization for ML: Only prefetch 1 task at a time 
# to prevent one worker from hogging 10 heavy training jobs.
celery_app.conf.worker_prefetch_multiplier = 1
celery_app.conf.task_track_started = True