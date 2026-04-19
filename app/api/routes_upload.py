

from __future__ import annotations
import os

import logging
import shutil
import uuid
from pathlib import Path
from typing import Annotated
from typing import List

import polars as pl
from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status


from app.schemas.upload_schema import UploadRequest, UploadResponse


from storage.s3_client import S3Client
from app.config import get_settings

settings = get_settings()

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["Dataset Management"])


TEMP_DIR = Path("tmp/uploads")
TEMP_DIR.mkdir(parents=True, exist_ok=True)


ALLOWED_EXTENSIONS = {".csv", ".parquet", ".xlsx", ".xls", ".xlsm"}


CONTENT_TYPE_MAP = {
    ".csv":     "text/csv",
    ".parquet": "application/octet-stream",
    ".xlsx":    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls":     "application/vnd.ms-excel",
    ".xlsm":    "application/vnd.ms-excel.sheet.macroenabled.12",
}



def get_s3_client() -> S3Client:
    """
    Provides an S3Client instance pointed at the datasets bucket.
    Reads credentials + bucket name from environment variables via config.
    Replace with a mock in tests.
    """

    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    if not access_key or not secret_key:
        raise ValueError("Missing AWS credentials")
    
    return S3Client(
        bucket_name  = os.getenv("S3_DATASETS_BUCKET", "ml-datasets"),
        endpoint_url = settings.s3.endpoint_url or None,      
        access_key=access_key,
        secret_key=secret_key,
        region       = os.getenv("AWS_REGION", "us-east-1"),
    )


def _safe_filename(filename: str) -> str:
    """
    Strip any directory components to prevent path traversal attacks.
    'uploads/../etc/passwd' → 'passwd'
    """
    return Path(filename).name


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload dataset(s)",
)
async def upload_dataset(
    files: List[UploadFile] = File(...),

    user_id: str = Form(...),
    file_types: List[str] = Form(...),
    file_sizes_mb: List[float] = Form(...),

    target_column: str | None = Form(default=None),
    problem_type: str | None = Form(default=None),

    # Merge config (optional)
    left_on: str | None = Form(default=None),
    right_on: str | None = Form(default=None),
    merge_how: str | None = Form(default=None),

    s3: S3Client = Depends(get_s3_client),
):
    """
    Supports:
      Single file upload
      Two-file merge workflow
    """

    
    if not (len(files) == len(file_types) == len(file_sizes_mb)):
        raise HTTPException(
            status_code=400,
            detail="files, file_types, and file_sizes_mb must have same length",
        )

    if len(files) > 4:
        raise HTTPException(
            status_code=400,
            detail="Maximum of 4 files allowed",
        )

   
    file_metas = []

    for i, file in enumerate(files):
        safe_name = _safe_filename(file.filename)
        extension = Path(safe_name).suffix.lower()

        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{extension}'",
            )

        file_metas.append({
            "filename": safe_name,
            "file_type": file_types[i],
            "file_size_mb": file_sizes_mb[i],
        })

   
    merge_config = None

    if len(files) == 2:
        if not left_on or not right_on:
            raise HTTPException(
                status_code=422,
                detail="Merge requires left_on and right_on",
            )

        merge_config = {
            "left_on": left_on,
            "right_on": right_on,
            "how": merge_how or "inner",
        }

    
    try:
        upload_meta = UploadRequest(
            files=file_metas,
            user_id=user_id,
            target_column=target_column,
            problem_type=problem_type,
            merge_config=merge_config,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    
    temp_paths = []
    s3_keys = []

    try:
        for i, file in enumerate(files):
            safe_name = upload_meta.files[i].filename
            s3_key = f"datasets/{user_id}/{uuid.uuid4().hex}_{safe_name}"

            temp_path = TEMP_DIR / f"{uuid.uuid4().hex}_{safe_name}"

            with temp_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            temp_paths.append(temp_path)
            s3_keys.append(s3_key)

        
        dataframes = []

        for path in temp_paths:
                ext = path.suffix.lower()

                if ext == ".csv":
                    
                    df = pl.read_csv(
                        path, 
                        infer_schema_length=10000,
                        try_parse_dates=True,
                        schema_overrides={"Time": pl.Float64},
                        
                        null_values=["", "NA", "null", "N/A"],
                        ignore_errors=False
                    )
                elif ext == ".parquet":
                    df = pl.read_parquet(path)
                else:
                
                    df = pl.read_excel(path, engine="fastexcel")

                dataframes.append(df)

                

            

        
        if len(dataframes) == 2:
            df = dataframes[0].join(
                dataframes[1],
                left_on=upload_meta.merge_config.left_on,
                right_on=upload_meta.merge_config.right_on,
                how=upload_meta.merge_config.how,
            )
            merged = True
        else:
            df = dataframes[0]
            merged = False

        dataset_shape = {"rows": df.height, "columns": df.width}

        
        final_key = f"datasets/{user_id}/{uuid.uuid4().hex}_final.parquet"

        df.write_parquet(temp_paths[0])  # reuse first path

        s3.upload_file(
            local_path=temp_paths[0],
            object_name=final_key,
            metadata={
                "user_id": user_id,
                "rows": str(dataset_shape["rows"]),
                "columns": str(dataset_shape["columns"]),
                "merged": str(merged),
            },
            content_type="application/octet-stream",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        for path in temp_paths:
            path.unlink(missing_ok=True)

    
    return UploadResponse(
        job_id=final_key,
        status="uploaded",
        message=f"Dataset processed successfully ({'merged' if merged else 'single file'})",
        file_count=len(files),
        merged=merged,
    )