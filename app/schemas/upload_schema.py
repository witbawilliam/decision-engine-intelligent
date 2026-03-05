
from __future__ import annotations

from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator



# Allowed Enums


AllowedFileType = Literal["csv", "parquet" "excel"]
AllowedProblemType = Literal["regression", "classification", "forecasting"]


# Upload Request Schema


class UploadRequest(BaseModel):
    """
    Validates dataset upload request metadata.
    """

    filename: str = Field(..., min_length=3)
    file_type: AllowedFileType
    file_size_mb: float = Field(..., gt=0)

    target_column: Optional[str] = None
    problem_type: Optional[AllowedProblemType] = None

    user_id: str = Field(..., min_length=3)

    
    # Validators
    

    @field_validator("file_size_mb")
    @classmethod
    def validate_file_size(cls, v: float):
        if v > 200:
            raise ValueError("File size exceeds 200MB limit.")
        return v

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v: str):
        if not ("." in v):
            raise ValueError("Filename must contain extension.")
        return v

    @field_validator("target_column")
    @classmethod
    def validate_target_column(cls, v):
        if v is not None and v.strip() == "":
            raise ValueError("Target column cannot be empty string.")
        return v



# Upload Response Schema

class UploadResponse(BaseModel):
    """
    Response returned after successful upload.
    """

    job_id: str
    status: Literal["uploaded", "queued", "excel"]
    message: str
