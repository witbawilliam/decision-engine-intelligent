from __future__ import annotations

import traceback
from datetime import datetime, timezone
from typing import Dict, Any

from monitoring.logging_config import get_logger

logger = get_logger(__name__)

class ErrorLogger:
   

    @staticmethod
    def log_error(
        component: str,
        error: Exception,
        context: Dict[str, Any] | None = None,
    ) -> None:
       
        
        payload = {
            "component": component,
            "error_type": type(error).__name__,
            "error_detail": str(error), 
            "traceback": traceback.format_exc(),
            "context": context or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        
        logger.error(
            f"Component '{component}' failed: {type(error).__name__}", 
            extra={"metadata": payload} 
        )