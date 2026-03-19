from __future__ import annotations

from typing import Optional, List, Literal, Annotated
from pydantic import BaseModel, Field, field_validator, model_validator



AllowedFileType = Literal["csv", "parquet", "excel"]
AllowedProblemType = Literal["regression", "classification", "forecasting"]
AllowedMergeStrategy = Literal["inner", "left", "right", "outer"]



class FileMeta(BaseModel):
    """
    Represents a single uploaded file.
    """

    filename: str = Field(..., min_length=3)
    file_type: AllowedFileType
    file_size_mb: float = Field(..., gt=0)

    @field_validator("file_size_mb")
    @classmethod
    def validate_file_size(cls, v: float):
        if v > 200:
            raise ValueError("File size exceeds 200MB limit.")
        return v

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v: str):
        if "." not in v:
            raise ValueError("Filename must contain extension.")
        return v



class MergeConfig(BaseModel):
    """
    Configuration used when merging two datasets.
    """

    left_on: str = Field(..., min_length=1)
    right_on: str = Field(..., min_length=1)
    how: AllowedMergeStrategy = "inner"



class UploadRequest(BaseModel):
    """
    Enterprise-level upload request supporting:
      Single dataset
      Dual dataset merge workflow
    """



    files: Annotated[List[FileMeta], Field(min_length=1, max_length=2)]
    target_column: Optional[str] = None
    problem_type: Optional[AllowedProblemType] = None
    merge_config: Optional[MergeConfig] = None
    user_id: str = Field(..., min_length=3)


    @field_validator("target_column")
    @classmethod
    def validate_target_column(cls, v):
        if v is not None and v.strip() == "":
            raise ValueError("Target column cannot be empty string.")
        return v


    @model_validator(mode="after")
    def validate_file_logic(self):
        file_count = len(self.files)

        # Case 1: Single file → OK, no merge needed
        if file_count == 1:
            if self.merge_config is not None:
                raise ValueError("Merge config should not be provided for single file upload.")

        # Case 2: Two files → must provide merge config
        elif file_count == 2:
            if self.merge_config is None:
                raise ValueError("Merge config is required when uploading two files.")

        return self



class UploadResponse(BaseModel):
    """
    Response returned after successful upload.
    """

    job_id: str
    status: Literal["uploaded", "queued", "processing", "complete", "faild"]
    message: str
    file_count: int
    merged: bool
