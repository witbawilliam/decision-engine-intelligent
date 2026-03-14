import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Standardized route imports
from app.api.routes_upload import router as upload_router
from app.api.routes_jobs import router as jobs_router
from app.api.routes_inference import router as inference_router
from app.api.routes_feedback import router as feedback_router

# --- 1. Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events.
    Ideal for warming up model caches or initializing DB pools.
    """
    logging.info(" Initializing Enterprise ML Platform...")
    # Startup logic here (e.g., registry.load_active_models())
    yield
    # Shutdown logic here (e.g., closing database connections)
    logging.info("Shutting down Enterprise ML Platform...")


# --- 2. Application Factory ---
app = FastAPI(
    title="Enterprise ML Platform",
    description="Scalable AutoML + Decision Intelligence System",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/v1/docs",      # Standardized documentation path
    redoc_url="/api/v1/redoc",
)

# --- 3. Middleware Stack ---
# Security: Define allowed origins clearly for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def monitor_performance(request: Request, call_next):
    """Observability: Tracks execution time for every request (SLA monitoring)."""
    start_time = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start_time
    response.headers["X-Response-Time"] = f"{duration:.4f}s"
    return response


# --- 4. Global Exception Mapping ---
@app.exception_handler(Exception)
async def universal_exception_handler(request: Request, exc: Exception):
    """
    Security: Prevents raw Python stack traces from leaking to the client.
    Maps all internal crashes to a clean, machine-readable JSON format.
    """
    logging.error(f"Unhandled system error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "message": "A critical error occurred. Trace ID has been logged.",
        },
    )


# --- 5. Route Registration ---
# We use v1 prefixing to support future API versions (V2, V3) without breaking clients.
API_V1_STR = "/api/v1"

app.include_router(upload_router, prefix=f"{API_V1_STR}/upload", tags=["Upload"])
app.include_router(jobs_router, prefix=f"{API_V1_STR}/jobs", tags=["Jobs"])
app.include_router(inference_router, prefix=f"{API_V1_STR}/inference", tags=["Inference"])
app.include_router(feedback_router, prefix=f"{API_V1_STR}/feedback", tags=["Feedback"])


# --- 6. Health & Readiness Probes ---
@app.get("/health", tags=["System"], status_code=status.HTTP_200_OK)
async def health_check():
    """Liveness probe for Kubernetes/Cloud load balancers."""
    return {
        "status": "online",
        "version": app.version,
        "timestamp": time.time()
    }

@app.get("/", include_in_schema=False)
def root():
    """Redirect or simple status for the base domain."""
    return {"platform": "Enterprise ML API", "docs": "/api/v1/docs"}