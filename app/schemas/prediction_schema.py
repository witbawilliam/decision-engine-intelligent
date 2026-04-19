from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)



_MAX_FEATURES          = 500          # Resilience: reject payload bombs
_MAX_FEATURE_KEY_LEN   = 128          # Resilience: guard against oversized keys
_MAX_BATCH_SIZE        = 1_000        # Performance: cap batch to prevent OOM
_MAX_MODEL_NAME_LEN    = 128
_MODEL_NAME_RE         = re.compile(r"^[a-zA-Z0-9_\-]+$")
_VERSION_RE            = re.compile(
    r"^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$|^latest$|^staging$|^production$"
)

explanations: Dict[str, Any] | None = None

class PipelineType(str, Enum):
    TEMPORAL    = "temporal"
    TABULAR     = "tabular"
    FORECASTING = "forecasting"


class PredictionStatus(str, Enum):
    SUCCESS = "success"
    FAILED  = "failed"
    TIMEOUT = "timeout"
    CACHED  = "cached"
    PARTIAL = "partial"     # some batch items failed, some succeeded


class ErrorCode(str, Enum):
    VALIDATION_ERROR    = "VALIDATION_ERROR"
    MODEL_NOT_FOUND     = "MODEL_NOT_FOUND"
    FEATURE_MISMATCH    = "FEATURE_MISMATCH"
    TIMEOUT             = "TIMEOUT"
    DRIFT_DETECTED      = "DRIFT_DETECTED"
    UPSTREAM_FAILURE    = "UPSTREAM_FAILURE"
    INTERNAL_ERROR      = "INTERNAL_ERROR"
    BATCH_SIZE_EXCEEDED = "BATCH_SIZE_EXCEEDED"
    RATE_LIMITED        = "RATE_LIMITED"



class TraceContext(BaseModel):
    """
    Distributed tracing envelope injected into every request and response.

    Carries trace_id and span_id across service boundaries so every log,
    metric, and audit record for a single inference can be correlated in
    Datadog / Jaeger / ELK without manual field plumbing.
    """

    model_config = ConfigDict(frozen=True)

    trace_id:    str = Field(default_factory=lambda: uuid4().hex)
    span_id:     str = Field(default_factory=lambda: uuid4().hex[:16])
    parent_span: Optional[str] = Field(
        default=None,
        description="Parent span for nested calls (e.g. pipeline → feature store)",
    )
    source: Optional[str] = Field(
        default=None,
        description="Originating service or component",
        examples=["api_gateway", "batch_worker", "celery_task"],
    )



class PredictionError(BaseModel):
    """
    Machine-readable error carrier attached to any failed response.

    Resilience: callers branch on error_code without parsing message strings,
    enabling automatic retry logic, circuit-breaker triggers, and alert routing.
    """

    model_config = ConfigDict(frozen=True)

    error_code:  ErrorCode
    message:     str
    retryable:   bool = Field(
        default=False,
        description="True if the same request may succeed on retry",
    )
    detail: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional structured context (e.g. which features were missing)",
    )
    occurred_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )



def compute_cache_key(
    model_name: str,
    version:    Optional[str],
    features:   Dict[str, Any],
) -> str:
    """
    Stable SHA-256 cache key from model identity + feature values.

    Uses json.dumps(sort_keys=True) for key-order independence and
    default=str for non-serialisable types.  Relies on hashlib (not
    Python's built-in hash()) so the key is identical across processes,
    restarts, and interpreter versions — safe to store in Redis.
    """
    payload = json.dumps(
        {"model": model_name, "version": version or "latest", "features": features},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()




class PredictionRequest(BaseModel):
    """
    Input payload for a single inference request.

    - model_name validated against a safe character allowlist (no injection).
    - version validated against semver / stage aliases; rejects arbitrary strings.
    - features hard-capped at 500 keys to prevent payload bombs.
    - All feature key lengths checked at ingestion.
    - extra="forbid" rejects undocumented fields at the API boundary.

    
    - TraceContext attached so every downstream log carries trace_id / span_id.
    - requested_at is timezone-aware UTC — no naive datetimes.

    - cache_key pre-computed once during validation via SHA-256.
      Downstream code reads self.cache_key directly — zero recomputation.
    """

    model_config = ConfigDict(extra="forbid")

    
    request_id: UUID = Field(
        default_factory=uuid4,
        description="Unique request ID — used for idempotency, audit, and dedup lock",
    )


    model_name: Annotated[str, Field(
        ...,
        min_length=1,
        max_length=_MAX_MODEL_NAME_LEN,
        description="Registered model name (alphanumeric, hyphens, underscores only)",
        examples=["sales_forecast_model", "churn-predictor-v2"],
    )]

    version: Optional[Annotated[str, Field(
        default=None,
        max_length=32,
        description="Semver (1.2.3) or stage alias (latest / staging / production)",
        examples=["1.2.3", "production", "latest"],
    )]] = None

    pipeline_type: PipelineType = Field(
        default=PipelineType.TABULAR,
        description="Pipeline to route this request through",
    )

    
    features: Dict[str, Any] = Field(
        ...,
        description="Feature dictionary for inference",
    )

    
    trace: TraceContext = Field(
        default_factory=TraceContext,
        description="Distributed tracing context propagated from the caller",
    )

    
    requested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the request was created",
    )

    
    cache_key: Optional[str] = Field(
        default=None,
        description="SHA-256 key derived from model + version + features. "
                    "Set automatically — do not supply manually.",
    )


    @field_validator("model_name")
    @classmethod
    def _validate_model_name(cls, v: str) -> str:
        if not _MODEL_NAME_RE.match(v):
            raise ValueError(
                f"model_name '{v}' contains invalid characters. "
                "Only alphanumerics, hyphens, and underscores are allowed."
            )
        return v

    @field_validator("version")
    @classmethod
    def _validate_version(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not _VERSION_RE.match(v):
            raise ValueError(
                f"version '{v}' is not a valid semver tag or stage alias. "
                "Expected e.g. '1.2.3', 'latest', 'staging', 'production'."
            )
        return v

    @field_validator("features")
    @classmethod
    def _validate_features(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if not v:
            raise ValueError("features must not be empty.")
        if len(v) > _MAX_FEATURES:
            raise ValueError(
                f"features exceeds the maximum of {_MAX_FEATURES} keys "
                f"(got {len(v)}). Split into smaller requests."
            )
        oversized = [k for k in v if len(str(k)) > _MAX_FEATURE_KEY_LEN]
        if oversized:
            raise ValueError(
                f"Feature key(s) exceed {_MAX_FEATURE_KEY_LEN} characters: "
                f"{oversized[:5]}"
            )
        return v

    @model_validator(mode="after")
    def _set_cache_key(self) -> "PredictionRequest":

        self.cache_key = compute_cache_key(self.model_name, self.version, self.features)
        return self



class PredictionResponse(BaseModel):
    """
    Output returned by a single inference call.


    - status enum makes success / failure machine-readable for dashboards.
    - trace mirrors the request context so logs correlate end-to-end.
    - cached flag feeds Redis hit-rate metrics without log parsing.
    - responded_at is timezone-aware UTC for accurate latency calculation.


    - error carries a PredictionError so callers distinguish model errors
      from infra failures without parsing message strings.
    - confidence validated to [0.0, 1.0] — rejects out-of-range model output.
    - frozen=True prevents accidental mutation after construction.
    """

    model_config = ConfigDict(frozen=True)

    request_id: UUID
    trace:      TraceContext

    
    model_name:    str
    model_version: str

    
    status:     PredictionStatus
    prediction: Optional[Any] = None

    confidence: Optional[Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        default=None,
        description="Model confidence score, constrained to [0.0, 1.0]",
    )

    
    error: Optional[PredictionError] = Field(
        default=None,
        description="Populated when status != success",
    )

    
    cached:     bool  = Field(default=False, description="True if served from Redis")
    latency_ms: float = Field(..., ge=0.0,   description="Wall-clock latency in ms")

    
    responded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dict for audit logging and API responses."""
        return self.model_dump(mode="json")



class BatchPredictionRequest(BaseModel):
    """
    Batch inference payload.

    
    - inputs hard-capped at _MAX_BATCH_SIZE (1 000) to prevent OOM.
    - Each item validated at ingestion so one bad row doesn't kill mid-batch.

    
    - priority field lets the scheduler fast-track urgent batches.
    - cache_keys property returns per-item SHA-256 keys so the service can
      issue a single Redis MGET for all items before touching the model.
    """

    model_config = ConfigDict(extra="forbid")

    request_id:    UUID = Field(default_factory=uuid4)
    model_name:    Annotated[str, Field(..., min_length=1, max_length=_MAX_MODEL_NAME_LEN)]
    version:       Optional[str] = None
    pipeline_type: PipelineType  = Field(default=PipelineType.TABULAR)

    inputs: Annotated[List[Dict[str, Any]], Field(
        ...,
        min_length=1,
        description="List of feature dicts — one per inference item",
    )]

    # Performance: higher-priority batches are dequeued first by the scheduler
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Scheduling priority 1 (lowest) 10 (highest)",
    )

    trace:        TraceContext = Field(default_factory=TraceContext)
    requested_at: datetime     = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


    @field_validator("model_name")
    @classmethod
    def _validate_model_name(cls, v: str) -> str:
        if not _MODEL_NAME_RE.match(v):
            raise ValueError(
                f"model_name '{v}' contains invalid characters."
            )
        return v

    @field_validator("version")
    @classmethod
    def _validate_version(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not _VERSION_RE.match(v):
            raise ValueError(
                f"version '{v}' is not a valid semver tag or stage alias."
            )
        return v

    @field_validator("inputs")
    @classmethod
    def _validate_inputs(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(v) > _MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(v)} exceeds the maximum of {_MAX_BATCH_SIZE}. "
                "Split into smaller batches."
            )
        for i, item in enumerate(v):
            if not item:
                raise ValueError(f"inputs[{i}] must not be empty.")
            if len(item) > _MAX_FEATURES:
                raise ValueError(
                    f"inputs[{i}] has {len(item)} features, exceeding the "
                    f"maximum of {_MAX_FEATURES}."
                )
        return v

    @property
    def cache_keys(self) -> List[str]:
        """
        Performance: one SHA-256 key per input item.
        Pass this list to RedisClient.mget() to resolve cache hits for the
        entire batch in a single round-trip before any model calls.
        """
        return [
            compute_cache_key(self.model_name, self.version, item)
            for item in self.inputs
        ]


class BatchItemResult(BaseModel):
    """
    Per-item result inside a BatchPredictionResponse.

    Resilience: individual items carry their own status and error so a single
    failed row does not suppress the predictions for the rest of the batch.
    """

    model_config = ConfigDict(frozen=True)

    index:      int                   # original position in inputs[]
    status:     PredictionStatus
    prediction: Optional[Any]  = None
    confidence: Optional[Annotated[float, Field(ge=0.0, le=1.0)]] = None
    cached:     bool           = False
    error:      Optional[PredictionError] = None



class BatchPredictionResponse(BaseModel):
    """
    Batch inference results.

    - overall status distinguishes SUCCESS / PARTIAL / FAILED at a glance.
    - Summary counters (total / succeeded / failed / cached) feed dashboards
      directly — no iteration required by the consumer.

    
    - results is List[BatchItemResult]: every item has its own status + error.
    - _validate_counts ensures summary counters are consistent with results,
      catching calculation bugs at construction time.

    - from_results() factory computes all counters in a single pass so callers
      never have to compute them manually.
    """

    model_config = ConfigDict(frozen=True)

    request_id:    UUID
    trace:         TraceContext
    model_name:    str
    model_version: str

    status:  PredictionStatus
    results: List[BatchItemResult]

    # Observability: summary counters for dashboards and alerting rules
    total:     int = Field(..., ge=0, description="Total items submitted")
    succeeded: int = Field(..., ge=0, description="Items that returned a prediction")
    failed:    int = Field(..., ge=0, description="Items that errored")
    cached:    int = Field(..., ge=0, description="Items served from Redis cache")

    latency_ms:   float    = Field(..., ge=0.0)
    responded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    

    @model_validator(mode="after")
    def _validate_counts(self) -> "BatchPredictionResponse":
        if self.succeeded + self.failed != self.total:
            raise ValueError(
                f"succeeded ({self.succeeded}) + failed ({self.failed}) "
                f"must equal total ({self.total})."
            )
        return self

    # Performance: factory 

    @classmethod
    def from_results(
        cls,
        *,
        request:       BatchPredictionRequest,
        results:       List[BatchItemResult],
        model_version: str,
        latency_ms:    float,
    ) -> "BatchPredictionResponse":
        """
        Build a BatchPredictionResponse from item results in one pass.

        Computes all summary counters automatically so callers never
        risk summary / results mismatches.

        Usage
        -----
            response = BatchPredictionResponse.from_results(
                request=batch_req,
                results=item_results,
                model_version="1.2.3",
                latency_ms=142.7,
            )
        """
        succeeded = sum(1 for r in results if r.status == PredictionStatus.SUCCESS)
        cached    = sum(1 for r in results if r.cached)
        failed    = len(results) - succeeded

        if failed == 0:
            overall = PredictionStatus.SUCCESS
        elif succeeded == 0:
            overall = PredictionStatus.FAILED
        else:
            overall = PredictionStatus.PARTIAL

        return cls(
            request_id    = request.request_id,
            trace         = request.trace,
            model_name    = request.model_name,
            model_version = model_version,
            status        = overall,
            results       = results,
            total         = len(results),
            succeeded     = succeeded,
            failed        = failed,
            cached        = cached,
            latency_ms    = latency_ms,
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")