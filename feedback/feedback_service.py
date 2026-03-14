from __future__ import annotations

from datetime import datetime
from typing import Dict, Any, List

from storage.postgres_client import PostgresClient
from monitoring.logging_config import get_logger

logger = get_logger(__name__)


class FeedbackService:
    """
    Collects prediction feedback for model improvement.

    Example:
        predicted price: 120
        real price: 150
    """

    TABLE_NAME = "model_feedback"


    @classmethod
    def store_feedback(
        cls,
        model_id: str,
        input_data: Dict[str, Any],
        prediction: Any,
        actual_value: Any,
    ) -> None:

        record = {
            "model_id": model_id,
            "input_data": input_data,
            "prediction": prediction,
            "actual_value": actual_value,
            "timestamp": datetime.utcnow(),
        }

        PostgresClient.insert(cls.TABLE_NAME, record)

        logger.info(
            "Feedback stored",
            extra={"model_id": model_id},
        )


    @classmethod
    def get_feedback(cls, model_id: str) -> List[Dict]:

        query = f"""
        SELECT *
        FROM {cls.TABLE_NAME}
        WHERE model_id = %s
        ORDER BY timestamp DESC
        """

        return PostgresClient.query(query, (model_id,))