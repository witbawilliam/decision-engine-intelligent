import time
import uuid
import logging

from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from monitoring.metrics import track_training_latency

logger = logging.getLogger("ml_platform")


class ProductionMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next) -> Response:

        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()
        request.state.request_id = request_id

        try:

            response = await call_next(request)
            

        except Exception as e:

    

            process_time = time.perf_counter() - start_time

            track_training_latency(
                task_id=request_id,
                status="failed",
                pipeline="inference",
                duration=process_time
            )

            logger.error(
                "unhandled_exception",
                exc_info=True,
                extra={
                    "request_id": request_id,
                    "path": request.url.path,
                    "method": request.method,
                    "latency": process_time,
                },
            )

            return JSONResponse(
                status_code=500,
                content={
                    "request_id": request_id,
                    "status": "failed",
                    "message": "Internal server error",
                },
            )

        process_time = time.perf_counter() - start_time

        track_training_latency(
            task_id=request_id,
            status="success",
            pipeline="inference",
            duration=process_time
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.4f}s"


        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "status_code": response.status_code,
                "latency": process_time,
            },
        )

        return response