import polars as pl
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks, status
from datetime import datetime, timezone
from typing import Literal

from core.pipelines.tabular_pipeline import TabularPipeline
from core.pipelines.temporal_pipeline import TemporalPipeline
from app.schemas.job_schema import V1

router = APIRouter(prefix="/v1/train", tags=["Model Training"])



def load_data(filename: str) -> pl.DataFrame:
    """
    Standardized data ingestion engine. 
    Supports CSV and Excel using high-performance Rust engines.
    """
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filename}")
        
    extension = path.suffix.lower()

    try:
        if extension == ".csv":
            return pl.read_csv(filename)
        
        elif extension in [".xlsx", ".xls", ".xlsm"]:
            
            return pl.read_excel(filename, engine="fastexcel")
        
        else:
            raise ValueError(f"Unsupported extension '{extension}'. Use .csv or .xlsx")
    except Exception as e:
        # In a real app, log this to Sentry or CloudWatch
        raise RuntimeError(f"Failed to parse {extension} file: {str(e)}")

# --- Background Worker ---

def run_training_task(job_params: V1.JobCreate, pipeline_type: Literal["tabular", "forecast"]):
    """
    Executes the heavy lifting. Decoupled from FastAPI for easy migration
    to Celery or Ray in the future.
    """
    try:
        
        df = load_data(job_params.filename)

        
        if pipeline_type == "tabular":
            pipeline = TabularPipeline(df=df, target_column=job_params.target_column)
        else:
            pipeline = TemporalPipeline(
                df=df, 
                target_column=job_params.target_column,
                time_column="ds", 
                model_type="prophet"
            )

        
        result = pipeline.run()
        print(f"Job {job_params.idempotency_key} Success: {result}")

    except Exception as e:
        print(f"Job {job_params.idempotency_key} Failed: {str(e)}")



@router.post("/tabular", status_code=status.HTTP_202_ACCEPTED, response_model=V1.JobStatusResponse)
async def train_tabular(request: V1.JobCreate, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_training_task, request, "tabular")
    return V1.JobStatusResponse(
        job_id=request.idempotency_key,
        status="queued",
        progress=0,
        updated_at=datetime.now(timezone.utc)
    )

@router.post("/forecast", status_code=status.HTTP_202_ACCEPTED, response_model=V1.JobStatusResponse)
async def train_forecast(request: V1.JobCreate, background_tasks: BackgroundTasks):
    if request.problem_type != "forecasting":
        raise HTTPException(status_code=400, detail="Problem type must be 'forecasting'")

    background_tasks.add_task(run_training_task, request, "forecast")
    return V1.JobStatusResponse(
        job_id=request.idempotency_key,
        status="queued",
        progress=0,
        updated_at=datetime.now(timezone.utc)
    )