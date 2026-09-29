import os
import tempfile
import uuid
from typing import List, Optional

import polars as pl
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from app.config import get_settings
from service.forecast_service import PredictionService, ForecastServiceError
from storage.s3_client import S3Client
from workers.tasks_forecast_inference import run_batch_forecast_task


router = APIRouter(prefix="/v1/forecast", tags=["Forecast Inference"])


class ForecastRequestSchema(BaseModel):
    periods: Optional[int] = None


class BatchForecastRequestSchema(BaseModel):
    forecast_horizon: Optional[int] = None
    idempotency_key: Optional[str] = None


class BatchForecastResponseSchema(BaseModel):
    job_id: str
    task_id: str
    status: str


# ── CSV-driven synchronous batch forecast ────────────────────────────────
#
# Deliberately separate from /batch (the Celery/all-production-models path).
# See module docstring on forecast_batch_from_csv() for why.

_MAX_SYNC_PRODUCTS = 100000  # HTTP-request safety cap -- see endpoint docstring


class BatchCsvForecastRequestSchema(BaseModel):
    s3_key: str
    product_id_column: str = "product_id"
    periods: Optional[int] = None


class ProductForecastResult(BaseModel):
    product_id: str
    status: str  # "success" | "failed"
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    periods: Optional[int] = None
    forecast: Optional[List[dict]] = None
    error: Optional[str] = None


class BatchCsvForecastResponseSchema(BaseModel):
    total_products: int
    succeeded: int
    failed: int
    results: List[ProductForecastResult]


def get_s3_client() -> S3Client:
    """
    Mirrors app/api/routes_upload.py's get_s3_client() so uploaded CSVs and
    this download path always resolve to the same bucket/credentials.
    """
    settings = get_settings()
    access_key = os.getenv("AWS_ACCESS_KEY_ID") or settings.s3.access_key
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY") or settings.s3.secret_key

    if not access_key or not secret_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="S3 credentials not configured.",
        )

    return S3Client(
        bucket_name=os.getenv("S3_DATASETS_BUCKET", settings.s3.bucket_name),
        endpoint_url=settings.s3.boto3_endpoint(),
        access_key=access_key,
        secret_key=secret_key,
        region=settings.s3.region,
    )


def get_forecast_service(request: Request) -> PredictionService:
    service = getattr(request.app.state, "forecast_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Forecast service not initialized.",
        )
    return service


@router.post("/predict/{product_id}", status_code=status.HTTP_200_OK)
async def forecast_product(
    product_id: str,
    payload: ForecastRequestSchema,
    service: PredictionService = Depends(get_forecast_service),
):
    try:
        result = service.run(product_id=product_id, periods=payload.periods)
        return {
            "product_id": result.product_id,
            "model_name": result.model_name,
            "model_version": result.model_version,
            "periods": result.periods,
            "generated_at": result.generated_at,
            "forecast": result.forecast.to_dict(orient="records"),
        }
    except ForecastServiceError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Forecast failed: {exc}")


@router.post(
    "/batch",
    response_model=BatchForecastResponseSchema,
    status_code=status.HTTP_202_ACCEPTED,
)
async def forecast_batch(payload: BatchForecastRequestSchema):
    """
    Triggers an async batch forecast across every product currently at
    stage='production'. Returns immediately with a job_id -- the actual
    work runs in a Celery worker via run_batch_forecast_task, since
    forecasting 100+ products sequentially would risk HTTP timeouts if
    done synchronously in the request/response cycle.
    """
    job_id = payload.idempotency_key or str(uuid.uuid4())

    job_payload = {
        "forecast_horizon": payload.forecast_horizon,
        "idempotency_key":  job_id,
    }

    try:
        async_result = run_batch_forecast_task.delay(job_payload)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to enqueue batch forecast task: {exc}",
        )

    return BatchForecastResponseSchema(
        job_id=job_id,
        task_id=async_result.id,
        status="queued",
    )


@router.post(
    "/batch-from-csv",
    response_model=BatchCsvForecastResponseSchema,
    status_code=status.HTTP_200_OK,
)
async def forecast_batch_from_csv(
    payload: BatchCsvForecastRequestSchema,
    service: PredictionService = Depends(get_forecast_service),
    s3: S3Client = Depends(get_s3_client),
):
    
    with tempfile.TemporaryDirectory() as tmp_dir:
       
        ext = os.path.splitext(payload.s3_key)[1].lower()
        local_path = os.path.join(tmp_dir, f"products{ext or '.csv'}")

        try:
            s3.download_file(object_name=payload.s3_key, local_path=local_path)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Could not download '{payload.s3_key}' from S3: {exc}",
            )

        try:
            if ext == ".parquet":
                df = pl.read_parquet(local_path)
            elif ext in (".xlsx", ".xls"):
                df = pl.read_excel(local_path)
            else:
                df = pl.read_csv(local_path)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Could not parse '{payload.s3_key}' as "
                    f"{ext.lstrip('.').upper() or 'CSV'}: {exc}"
                ),
            )

    if payload.product_id_column not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Column '{payload.product_id_column}' not found in uploaded file. "
                f"Available columns: {df.columns}"
            ),
        )

    product_ids = (
        df[payload.product_id_column]
        .cast(pl.Utf8, strict=False)
        .drop_nulls()
        .unique()
        .to_list()
    )

    if not product_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No product identifiers found in column '{payload.product_id_column}'.",
        )

    if len(product_ids) > _MAX_SYNC_PRODUCTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"{len(product_ids)} unique products found, exceeding the synchronous "
                f"limit of {_MAX_SYNC_PRODUCTS}. This endpoint is designed for small, "
                "curated lists -- for larger batches, the async /batch endpoint needs "
                "its ModelRegistry.save_forecast_results dependency fixed first."
            ),
        )

    results: List[ProductForecastResult] = []
    for product_id in product_ids:
        try:
            result = service.run(product_id=product_id, periods=payload.periods)
            results.append(
                ProductForecastResult(
                    product_id=product_id,
                    status="success",
                    model_name=result.model_name,
                    model_version=result.model_version,
                    periods=result.periods,
                    forecast=result.forecast.to_dict(orient="records"),
                )
            )
        except ForecastServiceError as exc:
            results.append(
                ProductForecastResult(product_id=product_id, status="failed", error=str(exc))
            )
        except Exception as exc:
            results.append(
                ProductForecastResult(
                    product_id=product_id, status="failed", error=f"Unexpected error: {exc}"
                )
            )

    succeeded = sum(1 for r in results if r.status == "success")
    return BatchCsvForecastResponseSchema(
        total_products=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
        results=results,
    )


@router.get("/health", status_code=status.HTTP_200_OK)
async def health(service: PredictionService = Depends(get_forecast_service)):
    return {"status": "healthy", "service": "forecast"}