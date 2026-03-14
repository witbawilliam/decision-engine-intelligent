from __future__ import annotations

import traceback
from datetime import datetime, timezone
from typing import Dict, Any

from monitoring.logging_config import get_logger

logger = get_logger(__name__)

class ErrorLogger:
    """
    Centralized, structured error logging for ML pipelines and inference.
    Solves the 'message' naming collision and provides standardized schemas.
    """

    @staticmethod
    def log_error(
        component: str,
        error: Exception,
        context: Dict[str, Any] | None = None,
    ) -> None:
        """
        Logs a structured error. Use 'context' for job-specific IDs.
        """
        # Nesting the payload prevents collisions with reserved 'message' or 'level' keys
        payload = {
            "component": component,
            "error_type": type(error).__name__,
            "error_detail": str(error), # Renamed from 'message' to avoid collision
            "traceback": traceback.format_exc(),
            "context": context or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Enterprise practice: Use a single key for all metadata to ensure
        # compatibility with ELK/Datadog/CloudWatch structured logging.
        logger.error(
            f"Component '{component}' failed: {type(error).__name__}", 
            extra={"metadata": payload} 
        )