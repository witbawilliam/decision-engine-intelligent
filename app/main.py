"""

What this file connects

config.py            get_settings() — loads all env vars once on startup
middleware.py        ProductionMiddleware — request ID, latency, error handling
health_checks.py     HealthChecker — liveness, readiness, startup, full report
routes_upload.py     /api/v1/datasets  — file upload + merge
routes_jobs.py       /api/v1/train     — training job dispatch via Celery
routes_inference.py  /api/v1/inference — real-time prediction via PredictionService
routes_feedback.py   /api/v1/feedback  — prediction feedback storage

Startup sequence

1. Settings validated  — missing env vars crash immediately with a clear error
2. Logging configured  — structured JSON logs from boot
3. FastAPI app created
4. Middleware attached  — ProductionMiddleware runs on every request
5. CORS configured
6. Routers registered  — all 4 routes active under /api/v1
7. Health endpoints registered — /health/live, /health/ready, /health/startup, /health/detail
8. lifespan startup    — HealthChecker.mark_startup_complete() called after boot

Run
───
    uvicorn main:app --reload                   # development
    uvicorn main:app --host 0.0.0.0 --port 8000 # production
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from app.config import get_settings
from app.middleware import ProductionMiddleware
from monitoring.health_checks import HealthChecker

from app.api.routes_upload    import router as upload_router
from app.api.routes_jobs      import router as jobs_router
from app.api.routes_inference import router as inference_router
from app.api.routes_feedback  import router as feedback_router



settings = get_settings()

logging.basicConfig(
    level   = settings.log_level,
    format  = "%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("ml_platform")

#  API prefix
API_V1 = "/api/v1"

# Internal API key header for /health/detail and /health/history
_internal_key_header = APIKeyHeader(name="X-Internal-Key", auto_error=False)



@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs ONCE on startup before the first request is accepted.
    Runs ONCE on shutdown after the last request completes.

    Startup
    ───────
    1. Log confirmed settings (no secrets — only names + environment)
    2. Mark HealthChecker startup complete so /health/ready returns healthy
       Add any warm-up logic here (e.g. load model into memory, warm Redis cache)

    Shutdown
    ────────
    Close any resources that need graceful teardown.
    """
    
    logger.info(
        "platform_starting",
        extra={
            "app_name":    settings.app_name,
            "environment": settings.environment,
            "log_level":   settings.log_level,
            "version":     "1.0.0",
        },
    )

    redis_ok    = HealthChecker.check_redis()["status"]    == "healthy"
    postgres_ok = HealthChecker.check_database()["status"] == "healthy"
    s3_ok       = HealthChecker.check_storage()["status"]  == "healthy"

    if not redis_ok:
        logger.warning("startup_warning: Redis is not reachable")
    if not postgres_ok:
        logger.warning("startup_warning: Postgres is not reachable")
    if not s3_ok:
        logger.warning("startup_warning: S3 is not reachable")

    HealthChecker.mark_startup_complete()
    logger.info("platform_ready", extra={"event": "startup_complete"})

    yield   # ← app runs here, accepting requests

    
    logger.info("platform_shutdown", extra={"event": "shutdown_initiated"})



app = FastAPI(
    title       = settings.app_name,
    description = (
        "Enterprise ML Platform — "
        "Upload data → Train model → Predict → Feedback"
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = f"{API_V1}/docs",
    redoc_url   = f"{API_V1}/redoc",
    openapi_url = f"{API_V1}/openapi.json",
)



# ProductionMiddleware — from middleware.py
# Adds X-Request-ID, X-Process-Time, structured logging, and catches
# unhandled exceptions so raw stack traces never reach the client
app.add_middleware(ProductionMiddleware)

# CORS — configure allowed origins per environment
_cors_origins = (
    ["*"]
    if settings.environment == "development"
    else [
        # Add your production domains here
        # "https://your-frontend.com",
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = _cors_origins,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)



@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Last-resort handler — catches anything ProductionMiddleware missed.
    Never leaks a stack trace to the client.
    request_id is read from request.state if ProductionMiddleware set it.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(
        "unhandled_exception",
        exc_info=True,
        extra={
            "request_id": request_id,
            "path":       request.url.path,
            "method":     request.method,
        },
    )
    return JSONResponse(
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
        content     = {
            "error":      "internal_server_error",
            "request_id": request_id,
            "message":    "An unexpected error occurred. The trace ID has been logged.",
        },
    )




app.include_router(upload_router)

app.include_router(jobs_router)

app.include_router(inference_router)

app.include_router(feedback_router)



@app.get(
    "/health/live",
    tags       = ["Health"],
    status_code= status.HTTP_200_OK,
    summary    = "Liveness probe",
    description= "Kubernetes liveness probe. Returns 200 if the process is alive.",
)
async def health_live():
    """
    HealthChecker.liveness_probe()
     is the process alive?
     only fails on catastrophic internal state
     Kubernetes restarts the pod if this returns non-200
    """
    return HealthChecker.liveness_probe()


@app.get(
    "/health/ready",
    tags       = ["Health"],
    status_code= status.HTTP_200_OK,
    summary    = "Readiness probe",
    description= "Kubernetes readiness probe. Checks Redis, Postgres, S3, and system resources.",
)
async def health_ready():
    """
    HealthChecker.readiness_probe()
    → checks Redis, Postgres, S3 via circuit breakers
    → checks CPU, memory, disk saturation
    → Kubernetes removes pod from load balancer if this fails
    """
    result = HealthChecker.readiness_probe()
    http_status = (
        status.HTTP_200_OK
        if result["status"] == "healthy"
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return JSONResponse(status_code=http_status, content=result)


@app.get(
    "/health/startup",
    tags       = ["Health"],
    status_code= status.HTTP_200_OK,
    summary    = "Startup probe",
    description= "Kubernetes startup probe. Returns healthy once lifespan startup completes.",
)
async def health_startup():
    """
    HealthChecker.startup_probe()
    → returns DEGRADED while model is still loading
    → returns HEALTHY once HealthChecker.mark_startup_complete() is called
    → prevents readiness/liveness checks from running too early
    """
    result = HealthChecker.startup_probe()
    http_status = (
        status.HTTP_200_OK
        if result["status"] in ("healthy", "degraded")
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return JSONResponse(status_code=http_status, content=result)


@app.get(
    "/health/detail",
    tags       = ["Health"],
    status_code= status.HTTP_200_OK,
    summary    = "Full health report (internal, API-key protected)",
    description= (
        "Complete health report including all dependencies, model artifact, "
        "system resources, and startup state. "
        "Requires X-Internal-Key header matching INTERNAL_HEALTH_API_KEY env var."
    ),
)
async def health_detail(
    api_key: Optional[str] = Security(_internal_key_header),
):
    """
    HealthChecker.full_report(api_key)
    → without valid key: returns public summary only (status + timestamp)
    → with valid key:    returns full internal report
      - redis latency + circuit breaker state
      - postgres latency + circuit breaker state
      - s3 latency + circuit breaker state
      - model artifact checksum + size validation
      - CPU / memory / disk / GPU resource saturation
      - startup state
    """
    return HealthChecker.full_report(api_key=api_key)


@app.get(
    "/health/history",
    tags       = ["Health"],
    status_code= status.HTTP_200_OK,
    summary    = "Health history (internal, API-key protected)",
    description= (
        "Rolling window of past health snapshots. "
        "Includes flapping detection. "
        "Requires X-Internal-Key header."
    ),
)
async def health_history(
    api_key: Optional[str] = Security(_internal_key_header),
):
    """
    HealthChecker.get_history(api_key)
     without valid key: returns {"authorized": False, "history": []}
     with valid key:    returns last N snapshots with flapping detection
    """
    return HealthChecker.get_history(api_key=api_key)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "platform": settings.app_name,
        "version":  "1.0.0",
        "docs":     f"{API_V1}/docs",
        "health":   "/health/live",
    }