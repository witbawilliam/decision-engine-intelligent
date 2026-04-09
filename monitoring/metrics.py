from __future__ import annotations
import functools
import hashlib
import json
import logging
import os
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from threading import Lock, RLock
from typing import Any, Callable, Dict, List, Optional, Tuple
from prometheus_client import Histogram

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
)


try:
    from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, Summary, push_to_gateway
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False

try:
    import mlflow  # type: ignore
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

logger = logging.getLogger("ml_platform.metrics")



TRAINING_LATENCY_HISTOGRAM = Histogram(
    'training_latency_seconds',
    'Time spent in training/inference pipelines',
    labelnames=['pipeline', 'status']
)

# A Counter for total jobs
TRAINING_JOB_COUNTER = Counter(
    'training_jobs_total',
    'Total number of jobs processed',
    labelnames=['pipeline', 'status']
)



class MetricsConfig:
    SERVICE_NAME:    str = os.getenv("SERVICE_NAME",    "ml-platform")
    ENVIRONMENT:     str = os.getenv("ENVIRONMENT",     "production")
    MODEL_VERSION:   str = os.getenv("MODEL_VERSION",   "unknown")
    DATASET_VERSION: str = os.getenv("DATASET_VERSION", "unknown")
    CI_CONFIDENCE:   float = float(os.getenv("CI_CONFIDENCE",   "0.95"))
    CI_N_BOOTSTRAP:  int   = int(os.getenv("CI_N_BOOTSTRAP",    "1000"))
    PSI_WARN_THRESHOLD:  float = float(os.getenv("PSI_WARN_THRESHOLD",  "0.1"))
    PSI_ALERT_THRESHOLD: float = float(os.getenv("PSI_ALERT_THRESHOLD", "0.2"))
    KS_ALPHA:            float = float(os.getenv("KS_ALPHA",            "0.05"))
    PROMETHEUS_GATEWAY:  str = os.getenv("PROMETHEUS_GATEWAY", "localhost:9091")
    PROMETHEUS_JOB:      str = os.getenv("PROMETHEUS_JOB",     "ml_metrics")
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT:   str = os.getenv("MLFLOW_EXPERIMENT",   "ml-platform")

    ALERT_THRESHOLDS: Dict[str, Tuple[float, float]] = {
        "accuracy":  (0.85, 0.75),
        "f1_score":  (0.80, 0.70),
        "auc_roc":   (0.80, 0.70),
        "mae":       (0.15, 0.30),
        "rmse":      (0.20, 0.40),
    }
    INVERTED_METRICS: set = {"mae", "rmse", "mse", "latency_p99_ms", "error_rate"}

class AlertSeverity(str, Enum):
    INFO, WARN, CRITICAL = "info", "warn", "critical"

class MetricMode(str, Enum):
    REALTIME, BATCH = "realtime", "batch"

@dataclass
class MetricResult:
    name: str; value: float; mode: MetricMode; model_version: str
    dataset_version: str; timestamp: str; run_id: str
    ci: Optional[Any] = None; tags: Dict[str, str] = field(default_factory=dict)
    alert: Optional[str] = None
    def to_dict(self) -> Dict: return asdict(self)



class MetricsRegistry:
    _counters:  Dict[str, int]   = defaultdict(int)
    _timings:   Dict[str, List[float]] = defaultdict(list)
    _lock = Lock()
    
    @classmethod
    def increment(cls, metric_name: str, value: int = 1, tags: Optional[Dict] = None) -> None:
        with cls._lock:
            cls._counters[metric_name] += value
        logger.debug(f"Metric {metric_name} incremented by {value}")



def track_training_latency(task_id: str, status: str, pipeline: str, duration: float = None):
    """Bridge for worker telemetry."""
    try:
        # 1. Update your custom Registry (The dict-based one you have)
        metric_name = f"{pipeline}_job_{status}"
        MetricsRegistry.increment(metric_name, tags={"task_id": task_id})

        # 2. Update Prometheus (This allows .observe() and .inc() to work)
        if _PROMETHEUS_AVAILABLE:
            TRAINING_JOB_COUNTER.labels(pipeline=pipeline, status=status).inc()
            if duration is not None:
                TRAINING_LATENCY_HISTOGRAM.labels(pipeline=pipeline, status=status).observe(duration)

    except Exception as e:
        logger.error(f"Failed to record telemetry: {e}")
    
    # Structured Logging
    logger.info("telemetry_recorded", extra={
        "task_id": task_id, 
        "status": status, 
        "pipeline": pipeline, 
        "duration": duration,
        "metric_name": metric_name
    })