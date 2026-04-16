
from __future__ import annotations
import re
from datetime import datetime, timezone
from typing import Annotated, Literal, Optional

from pydantic import Field, StringConstraints
from pydantic import BaseModel
from typing import Literal, Optional, Any




UUIDStr = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    ),
]


ModelVersionStr = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$",
        max_length=32,
    ),
]


IdempotencyKeyStr = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^[\x21-\x7E]{16,128}$",   # printable ASCII, no space/control chars
    ),
]

# Progress percentage [0, 100].
ProgressInt = Annotated[int, Field(ge=0, le=100)]

# Normalised score / metric in [0.0, 1.0].
NormalisedFloat = Annotated[float, Field(ge=0.0, le=1.0)]



JobStatus = Literal[
    "uploaded",
    "validated",
    "queued",
    "running",
    "completed",
    "failed",
    "cancelled",
]

ProblemType = Literal["regression", "classification", "forecasting"]



TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "failed", "cancelled"})

ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "uploaded":  frozenset({"validated", "failed", "cancelled"}),
    "validated": frozenset({"queued",    "failed", "cancelled"}),
    "queued":    frozenset({"running",   "failed", "cancelled"}),
    "running":   frozenset({"completed", "failed", "cancelled"}),
    "completed": frozenset(),
    "failed":    frozenset(),
    "cancelled": frozenset(),
}



_SAFE_FILENAME_RE = re.compile(r"^[\w\- ]+\.[a-zA-Z0-9]{1,10}$")
_COLUMN_NAME_RE   = re.compile(r"^\w+$")

class V1:

    class JobCreate(BaseModel):
        user_id: UUIDStr
        idempotency_key: IdempotencyKeyStr
        filename: str
        target_column: Optional[str] = None
        problem_type: Optional[ProblemType] = None
        s3_key: Optional[str] = None            # ← S3 path to the uploaded dataset
        reference_s3_key: Optional[str] = None

    class JobStatusResponse(BaseModel):
        """Used for training updates"""
        job_id: IdempotencyKeyStr
        status: JobStatus
        progress: ProgressInt
        updated_at: datetime

    class JobResultResponse(BaseModel):
        """Used for inference/prediction results"""
        job_id: IdempotencyKeyStr
        status: Literal["completed"]
        model_version: ModelVersionStr
        performance: dict
        insight_summary: Optional[str] = None

    

    class FeedbackResponse(BaseModel):
        status: str = "success"
        message: str
    

    @staticmethod
    def validate_filename(v: str) -> str:
        
        pass

    class FeedbackRequest(BaseModel):
        model_name: str
        prediction: float
        actual: float
        metadata: Optional[dict[str, Any]] = None

    class FeedbackResponse(BaseModel):
        feedback_id: str
        model_name: str
        prediction: float
        actual: float
        absolute_error: float
        squared_error: float
        relative_error: Optional[float]
        recorded_at: datetime
        metadata: dict[str, Any]
    
    def validate_filename(v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("filename must not be empty.")
        if not _SAFE_FILENAME_RE.match(v):
            raise ValueError(
                f"filename '{v}' contains disallowed characters. "
                "Only alphanumerics, hyphens, underscores, spaces, and a "
                "single dot-separated extension are permitted."
            )
        if len(v) > 255:
            raise ValueError("filename must not exceed 255 characters.")
        return v


    def validate_column_name(v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip()
        if not v:
            raise ValueError("target_column must not be an empty string.")
        if not _COLUMN_NAME_RE.match(v):
            raise ValueError(
                f"target_column '{v}' must contain only word characters (a-z, A-Z, 0-9, _)."
            )
        if len(v) > 128:
            raise ValueError("target_column must not exceed 128 characters.")
        return v


    def ensure_utc(v: datetime) -> datetime:
        """Reject naïve datetimes; normalise tz-aware values to UTC."""
        if v.tzinfo is None:
            raise ValueError("Datetimes must be timezone-aware (UTC).")
        return v.astimezone(timezone.utc)
