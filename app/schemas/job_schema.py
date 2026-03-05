from __future__ import annotations

from typing import Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field

# Enums

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


# Job Creation Schema (Internal Use)

class JobCreate(BaseModel):
    """
    Internal schema used when a new ML job is created.
    """

    user_id: str = Field(..., min_length=3)
    filename: str
    target_column: Optional[str] = None
    problem_type: Optional[ProblemType] = None

# Job Metadata Schema (Stored in PostgreSQL)

class JobMetadata(BaseModel):
    """
    Represents full metadata stored for each ML job.
    """

    job_id: str
    user_id: str
    filename: str

    status: JobStatus

    problem_type: Optional[ProblemType] = None
    target_column: Optional[str] = None

    quality_score: Optional[float] = None
    model_version: Optional[str] = None

    error_message: Optional[str] = None

    created_at: datetime
    updated_at: datetime


# Job Status Response (API)

class JobStatusResponse(BaseModel):
    """
    API response when querying job status.
    """

    job_id: str
    status: JobStatus
    progress: int = Field(..., ge=0, le=100)

    message: Optional[str] = None


# Job Result Response

class JobResultResponse(BaseModel):
    """
    Returned when job completes successfully.
    """

    job_id: str
    status: Literal["completed"]

    model_version: str
    performance_metric: float
    insight_summary: Optional[str] = None
