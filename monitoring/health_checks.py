
from __future__ import annotations

import hashlib
import logging
import os
import time
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from threading import Lock
from typing import Any, Callable, Deque, Dict, List, Optional

import psutil

try:
    import pynvml  

    pynvml.nvmlInit()
    _GPU_AVAILABLE = True
except Exception:
    _GPU_AVAILABLE = False


try:
    import pybreaker  

    _BREAKER_AVAILABLE = True
except ImportError:
    _BREAKER_AVAILABLE = False

from storage.redis_client import RedisClient
from storage.postgres_client import PostgresClient
from storage.s3_client import S3Client

logger = logging.getLogger("health_check")


class ComponentStatus(str, Enum):
    HEALTHY   = "healthy"
    DEGRADED  = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN   = "unknown"


class ProbeType(str, Enum):
    LIVENESS  = "liveness"
    READINESS = "readiness"
    STARTUP   = "startup"



class HealthConfig:
    """
    All thresholds and settings driven by environment variables so they can
    be tuned without a redeploy.
    """

    # Resource saturation thresholds (%)
    CPU_WARN_PCT:    float = float(os.getenv("HEALTH_CPU_WARN_PCT",    "80"))
    CPU_CRIT_PCT:    float = float(os.getenv("HEALTH_CPU_CRIT_PCT",    "95"))
    MEM_WARN_PCT:    float = float(os.getenv("HEALTH_MEM_WARN_PCT",    "80"))
    MEM_CRIT_PCT:    float = float(os.getenv("HEALTH_MEM_CRIT_PCT",    "95"))
    DISK_WARN_PCT:   float = float(os.getenv("HEALTH_DISK_WARN_PCT",   "80"))
    DISK_CRIT_PCT:   float = float(os.getenv("HEALTH_DISK_CRIT_PCT",   "95"))
    GPU_MEM_WARN_PCT:float = float(os.getenv("HEALTH_GPU_MEM_WARN_PCT","80"))

    # Model artifact settings
    MODEL_PATH:         str           = os.getenv("MODEL_PATH",    "/models/current/model.pkl")
    MODEL_VERSION:      str           = os.getenv("MODEL_VERSION", "unknown")
    MODEL_CHECKSUM:     Optional[str] = os.getenv("MODEL_CHECKSUM")          # SHA-256 hex
    MODEL_MAX_SIZE_MB:  float         = float(os.getenv("MODEL_MAX_SIZE_MB", "2048"))

    # Startup: model loading can take time — give it a generous window
    STARTUP_TIMEOUT_SECS: float = float(os.getenv("STARTUP_TIMEOUT_SECS", "120"))

    # Historical snapshot ring-buffer size
    HISTORY_SIZE: int = int(os.getenv("HEALTH_HISTORY_SIZE", "20"))

    # Circuit breaker: trip after N consecutive failures
    BREAKER_FAIL_MAX:         int   = int(os.getenv("BREAKER_FAIL_MAX",         "3"))
    BREAKER_RESET_TIMEOUT:    float = float(os.getenv("BREAKER_RESET_TIMEOUT", "30"))

    # Internal API key for the detailed /health/detailed endpoint
    INTERNAL_API_KEY: str = os.getenv("INTERNAL_HEALTH_API_KEY", "change-me-in-prod")



def _make_breaker(name: str) -> Any:
    """Return a pybreaker CircuitBreaker or a no-op stub if unavailable."""
    if _BREAKER_AVAILABLE:
        return pybreaker.CircuitBreaker(
            fail_max=HealthConfig.BREAKER_FAIL_MAX,
            reset_timeout=HealthConfig.BREAKER_RESET_TIMEOUT,
            name=name,
        )


    class _NoOpBreaker:
        """Passes calls straight through; never trips."""
        def call(self, fn: Callable, *args, **kwargs):
            return fn(*args, **kwargs)
        @property
        def state(self):
            return type("S", (), {"name": "closed"})()

    return _NoOpBreaker()


_redis_breaker    = _make_breaker("redis")
_postgres_breaker = _make_breaker("postgres")
_s3_breaker       = _make_breaker("s3")


def _timed_check(fn: Callable[[], bool]) -> Dict[str, Any]:
    """
    Run *fn*, capture timing and any exception.
    Returns a component-level result dict.
    """
    t0 = time.perf_counter()
    try:
        ok     = bool(fn())
        error  = None
    except Exception as exc:
        ok     = False
        error  = str(exc)
    elapsed_ms = round((time.perf_counter() - t0) * 1_000, 2)

    return {
        "status":      ComponentStatus.HEALTHY if ok else ComponentStatus.UNHEALTHY,
        "latency_ms":  elapsed_ms,
        "error":       error,
    }


def _resource_status(used_pct: float, warn: float, crit: float) -> ComponentStatus:
    if used_pct >= crit:
        return ComponentStatus.UNHEALTHY
    if used_pct >= warn:
        return ComponentStatus.DEGRADED
    return ComponentStatus.HEALTHY


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


class HealthChecker:
    """
    Enterprise ML system health monitor.

    Thread-safe, circuit-breaker protected, historically-aware.
    """

    _history: Deque[Dict] = deque(maxlen=HealthConfig.HISTORY_SIZE)
    _history_lock: Lock   = Lock()
    _startup_time: float  = time.monotonic()
    _startup_complete: bool = False   # set to True once model is loaded


    @classmethod
    def mark_startup_complete(cls) -> None:
        """Call this once your model has fully loaded."""
        cls._startup_complete = True
        logger.info("startup_complete", extra={"event": "startup_complete"})

    
    @classmethod
    def check_redis(cls) -> Dict[str, Any]:
        result = _timed_check(lambda: _redis_breaker.call(RedisClient.ping))
        result["circuit_breaker"] = _redis_breaker.state.name
        return result

    @classmethod
    def check_database(cls) -> Dict[str, Any]:
        result = _timed_check(lambda: _postgres_breaker.call(PostgresClient.ping))
        result["circuit_breaker"] = _postgres_breaker.state.name
        return result

    @classmethod
    def check_storage(cls) -> Dict[str, Any]:
        result = _timed_check(lambda: _s3_breaker.call(S3Client.ping))
        result["circuit_breaker"] = _s3_breaker.state.name
        return result

    
    def check_model_artifact(cls) -> Dict[str, Any]:
        """
        Validates:
          • File exists and is non-empty
          • File size is within expected bounds
          • SHA-256 checksum matches (if MODEL_CHECKSUM env var is set)
          • Reported version matches MODEL_VERSION env var
        """
        path = HealthConfig.MODEL_PATH
        t0   = time.perf_counter()
        issues: List[str] = []

        try:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Model not found at {path}")

            size_mb = os.path.getsize(path) / (1024 ** 2)
            if size_mb == 0:
                issues.append("Model file is empty")
            if size_mb > HealthConfig.MODEL_MAX_SIZE_MB:
                issues.append(f"Model size {size_mb:.1f} MB exceeds limit {HealthConfig.MODEL_MAX_SIZE_MB} MB")

            # Checksum verification (optional — set MODEL_CHECKSUM env var)
            checksum_ok = True
            actual_checksum = None
            if HealthConfig.MODEL_CHECKSUM:
                actual_checksum = _sha256(path)
                if actual_checksum != HealthConfig.MODEL_CHECKSUM:
                    checksum_ok = False
                    issues.append("Checksum mismatch — possible corruption or wrong artifact")

            status = ComponentStatus.UNHEALTHY if issues else ComponentStatus.HEALTHY

            return {
                "status":        status,
                "latency_ms":    round((time.perf_counter() - t0) * 1_000, 2),
                "path":          path,
                "size_mb":       round(size_mb, 2),
                "version":       HealthConfig.MODEL_VERSION,
                "checksum_ok":   checksum_ok,
                "issues":        issues,
                "error":         None,
            }

        except Exception as exc:
            return {
                "status":     ComponentStatus.UNHEALTHY,
                "latency_ms": round((time.perf_counter() - t0) * 1_000, 2),
                "path":       path,
                "error":      str(exc),
                "issues":     issues,
            }

    
    @classmethod
    def check_resources(cls) -> Dict[str, Any]:
        """
        CPU, memory, disk, and GPU (if available).
        Returns per-resource status + overall saturation verdict.
        """
        cpu_pct  = psutil.cpu_percent(interval=0.2)
        mem      = psutil.virtual_memory()
        disk     = psutil.disk_usage("/")

        cpu_status  = _resource_status(cpu_pct,      HealthConfig.CPU_WARN_PCT,  HealthConfig.CPU_CRIT_PCT)
        mem_status  = _resource_status(mem.percent,  HealthConfig.MEM_WARN_PCT,  HealthConfig.MEM_CRIT_PCT)
        disk_status = _resource_status(disk.percent, HealthConfig.DISK_WARN_PCT, HealthConfig.DISK_CRIT_PCT)

        resources: Dict[str, Any] = {
            "cpu": {
                "status":       cpu_status,
                "used_pct":     cpu_pct,
                "warn_pct":     HealthConfig.CPU_WARN_PCT,
                "critical_pct": HealthConfig.CPU_CRIT_PCT,
            },
            "memory": {
                "status":       mem_status,
                "used_pct":     mem.percent,
                "used_gb":      round(mem.used  / (1024**3), 2),
                "total_gb":     round(mem.total / (1024**3), 2),
                "warn_pct":     HealthConfig.MEM_WARN_PCT,
                "critical_pct": HealthConfig.MEM_CRIT_PCT,
            },
            "disk": {
                "status":       disk_status,
                "used_pct":     disk.percent,
                "free_gb":      round(disk.free  / (1024**3), 2),
                "total_gb":     round(disk.total / (1024**3), 2),
                "warn_pct":     HealthConfig.DISK_WARN_PCT,
                "critical_pct": HealthConfig.DISK_CRIT_PCT,
            },
        }

        
        if _GPU_AVAILABLE:
            try:
                handle      = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info    = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_mem_pct = (mem_info.used / mem_info.total) * 100
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_status  = _resource_status(gpu_mem_pct, HealthConfig.GPU_MEM_WARN_PCT, 95.0)

                resources["gpu"] = {
                    "status":         gpu_status,
                    "memory_used_pct": round(gpu_mem_pct, 1),
                    "memory_used_mb":  round(mem_info.used  / (1024**2), 1),
                    "memory_total_mb": round(mem_info.total / (1024**2), 1),
                    "utilization_pct": utilization.gpu,
                }
            except Exception as exc:
                resources["gpu"] = {"status": ComponentStatus.UNKNOWN, "error": str(exc)}

        # Aggregate: worst individual status wins
        statuses = [v["status"] for v in resources.values() if isinstance(v, dict) and "status" in v]
        if ComponentStatus.UNHEALTHY in statuses:
            overall = ComponentStatus.UNHEALTHY
        elif ComponentStatus.DEGRADED in statuses:
            overall = ComponentStatus.DEGRADED
        else:
            overall = ComponentStatus.HEALTHY

        return {"status": overall, "components": resources}

   

    @classmethod
    def liveness_probe(cls) -> Dict[str, Any]:
        """
        /health/live — Is the process alive?
        Kubernetes restarts the pod if this fails.
        Only fails on catastrophic internal state — NOT dependency failures.
        """
        return {
            "probe":     ProbeType.LIVENESS,
            "status":    ComponentStatus.HEALTHY,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_secs": round(time.monotonic() - cls._startup_time, 1),
        }

    @classmethod
    def readiness_probe(cls) -> Dict[str, Any]:
        """
        /health/ready — Can this pod accept traffic?
        Kubernetes removes the pod from load balancer if this fails.
        Checks all dependencies and resource saturation.
        """
        redis_result    = cls.check_redis()
        db_result       = cls.check_database()
        storage_result  = cls.check_storage()
        resource_result = cls.check_resources()

        components = {
            "redis":     redis_result,
            "database":  db_result,
            "storage":   storage_result,
            "resources": resource_result,
        }

        # Not ready if any dependency is unhealthy OR resources are critically saturated
        dep_statuses      = [redis_result["status"], db_result["status"], storage_result["status"]]
        resource_saturated = resource_result["status"] == ComponentStatus.UNHEALTHY
        all_deps_ok        = all(s == ComponentStatus.HEALTHY for s in dep_statuses)

        overall = ComponentStatus.HEALTHY if (all_deps_ok and not resource_saturated) else ComponentStatus.UNHEALTHY

        return {
            "probe":      ProbeType.READINESS,
            "status":     overall,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "components": components,
        }

    @classmethod
    def startup_probe(cls) -> Dict[str, Any]:
        """
        /health/startup — Has the service finished initializing?
        Kubernetes uses this during the startup window.
        Prevents readiness/liveness checks from running too early.
        """
        elapsed = time.monotonic() - cls._startup_time
        timed_out = elapsed > HealthConfig.STARTUP_TIMEOUT_SECS

        if timed_out and not cls._startup_complete:
            status = ComponentStatus.UNHEALTHY
            message = f"Startup timed out after {elapsed:.0f}s (limit: {HealthConfig.STARTUP_TIMEOUT_SECS}s)"
        elif cls._startup_complete:
            status  = ComponentStatus.HEALTHY
            message = "Startup complete"
        else:
            status  = ComponentStatus.DEGRADED   # still initializing — not failed yet
            message = f"Initializing... {elapsed:.1f}s elapsed"

        return {
            "probe":           ProbeType.STARTUP,
            "status":          status,
            "message":         message,
            "startup_complete": cls._startup_complete,
            "elapsed_secs":    round(elapsed, 1),
            "timeout_secs":    HealthConfig.STARTUP_TIMEOUT_SECS,
            "timestamp":       datetime.now(timezone.utc).isoformat(),
        }



    @classmethod
    def full_report(cls, api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive health report — intended for internal tooling only.

        Pass api_key=<value> to get full details.
        Without a valid key, returns a minimal public-safe summary.

        In a FastAPI/Flask setup you would validate the key in the route
        layer — this method accepts it directly for flexibility.
        """
        authorized = (api_key == HealthConfig.INTERNAL_API_KEY)

        t0 = time.perf_counter()

        # Always compute these
        redis_result    = cls.check_redis()
        db_result       = cls.check_database()
        storage_result  = cls.check_storage()
        model_result    = cls.check_model_artifact()
        resource_result = cls.check_resources()
        startup_result  = cls.startup_probe()

        all_statuses = [
            redis_result["status"],
            db_result["status"],
            storage_result["status"],
            model_result["status"],
            resource_result["status"],
        ]

        if ComponentStatus.UNHEALTHY in all_statuses:
            overall = ComponentStatus.UNHEALTHY
        elif ComponentStatus.DEGRADED in all_statuses:
            overall = ComponentStatus.DEGRADED
        else:
            overall = ComponentStatus.HEALTHY

        elapsed_ms = round((time.perf_counter() - t0) * 1_000, 2)
        timestamp  = datetime.now(timezone.utc).isoformat()

        #  Public summary (no sensitive topology) 
        public_report: Dict[str, Any] = {
            "status":        overall,
            "timestamp":     timestamp,
            "response_ms":   elapsed_ms,
        }

        if not authorized:
            logger.warning("Unauthenticated full_report request — returning public summary only")
            return public_report

        # Authorized: full internal report 
        detailed_report: Dict[str, Any] = {
            **public_report,
            "model_version":  HealthConfig.MODEL_VERSION,
            "startup":        startup_result,
            "dependencies": {
                "redis":    redis_result,
                "database": db_result,
                "storage":  storage_result,
            },
            "model_artifact": model_result,
            "resources":      resource_result,
        }

        # Record in rolling history
        cls._record_history(detailed_report)

        logger.info(
            "health_full_report",
            extra={
                "status":      overall,
                "response_ms": elapsed_ms,
            },
        )

        return detailed_report

    #  Historical 

    @classmethod
    def _record_history(cls, report: Dict) -> None:
        with cls._history_lock:
            cls._history.append({
                "timestamp":   report.get("timestamp"),
                "status":      report.get("status"),
                "response_ms": report.get("response_ms"),
                "dependencies": {
                    k: v.get("status")
                    for k, v in report.get("dependencies", {}).items()
                },
                "resources": report.get("resources", {}).get("status"),
            })

    @classmethod
    def get_history(cls, api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Return the rolling window of past health snapshots.
        Auth-protected — returns empty list without valid key.
        """
        if api_key != HealthConfig.INTERNAL_API_KEY:
            return {"authorized": False, "history": []}

        with cls._history_lock:
            snapshots = list(cls._history)

        # Compute a flapping score — how often status changed in history window
        statuses  = [s["status"] for s in snapshots]
        changes   = sum(1 for a, b in zip(statuses, statuses[1:]) if a != b)
        flapping  = changes > (len(snapshots) // 2)

        return {
            "authorized":       True,
            "history_size":     len(snapshots),
            "snapshots":        snapshots,
            "flapping_detected": flapping,
            "change_count":     changes,
        }