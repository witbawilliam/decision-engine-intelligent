import os
import shutil
import polars as pl
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from typing import Dict, Any

from app.schemas.job_schema import V1 

router = APIRouter(prefix="/v1/datasets", tags=["Dataset Management"])

UPLOAD_FOLDER = Path("datasets")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".xlsm", ".parquet"}

def get_safe_path(filename: str) -> Path:
    """Prevents path traversal attacks by extracting only the filename."""
    safe_name = Path(filename).name 
    return UPLOAD_FOLDER / safe_name

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Enterprise-grade file upload with streaming, security, and metadata extraction.
    """
    file_extension = Path(file.filename).suffix.lower()
    
    #  Early Validation
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}"
        )

    file_path = get_safe_path(file.filename)

    
    try:
        with file_path.open("wb") as buffer:
            # Iterates in chunks (default 1MB) so we don't load 2GB into RAM
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Disk I/O failure: {str(e)}"
        )
    finally:
        file.file.close()

    #  Metadata Extraction with Polars
    try:
        if file_extension == ".csv":
            df = pl.read_csv(file_path)
        elif file_extension == ".parquet":
            df = pl.read_parquet(file_path)
        else:
            # Excel support
            df = pl.read_excel(file_path, engine="fastexcel")

        return {
            "filename": file_path.name,
            "storage_path": str(file_path),
            "size_bytes": file_path.stat().st_size,
            "shape": {
                "rows": df.height,
                "columns": df.width
            },
            "schema": {name: str(dtype) for name, dtype in df.schema.items()},
            "created_at": str(pl.datetime.now())
        }
        
    except Exception as e:
        # If the file is corrupted or unreadable, delete the garbage file
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"File uploaded but could not be parsed: {str(e)}"
        )