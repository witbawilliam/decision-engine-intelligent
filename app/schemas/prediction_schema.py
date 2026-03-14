

from __future__ import annotations

from datetime import datetime
from typing import Dict, Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict


class PredictionRequest(BaseModel):
    """
    Input payload sent to the ML service.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: UUID = Field(default_factory=uuid4)

    model_name: str = Field(
        ...,
        description="Name of the registered model",
        examples=["sales_forecast_model"],
    )

    version: Optional[str] = Field(
        default=None,
        description="Optional model version",
    )

    features: Dict[str, Any] = Field(
        ...,
        description="Feature dictionary used for prediction",
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Prediction request timestamp",
    )


class PredictionResponse(BaseModel):
    """
    Output returned by the prediction service.
    """

    request_id: UUID
    model_name: str
    model_version: str

    prediction: Any

    confidence: Optional[float] = Field(
        default=None,
        description="Optional confidence score",
    )

    latency_ms: float = Field(
        ...,
        description="Prediction latency in milliseconds",
    )

    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BatchPredictionRequest(BaseModel):
    """
    Batch inference payload.
    """

    model_name: str
    version: Optional[str] = None

    inputs: list[Dict[str, Any]]

    request_id: UUID = Field(default_factory=uuid4)


class BatchPredictionResponse(BaseModel):
    """
    Batch inference results.
    """

    request_id: UUID
    predictions: list[Any]
    latency_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)