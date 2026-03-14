"""
services/prediction_service.py

Central prediction orchestration service.
"""

import time
from typing import Any, Dict

from core.pipelines.temporal_pipeline import TemporalPipeline
from app.schemas.prediction_schema import PredictionRequest, PredictionResponse
from monitoring.metrics import MetricsCollector
from monitoring.logging_config import get_logger
from feedback.feedback_service import FeedbackService
from app.dependencies import get_s3_client, get_redis_client


logger = get_logger(__name__)


class PredictionService:
    """
    Service responsible for executing the ML prediction pipeline.
    """

    def __init__(self):

        self.pipeline = TemporalPipeline()

        self.metrics = MetricsCollector()

        self.feedback = FeedbackService()

        self.redis = get_redis_client()
        self.storage = get_s3_client()


    def predict(self, request: PredictionRequest) -> PredictionResponse:

        start = time.time()

        try:

            logger.info(f"Prediction request received: {request.request_id}")

            # Run pipeline
            prediction = self.pipeline.predict(request.features)

            latency = (time.time() - start) * 1000

            # monitoring metrics
            self.metrics.record_prediction_latency(latency)

            response = PredictionResponse(
                request_id=request.request_id,
                model_name=request.model_name,
                model_version="latest",
                prediction=prediction,
                latency_ms=latency,
            )

            return response

        except Exception as e:

            self.feedback.log_error(str(e))

            logger.exception("Prediction failed")

            raise