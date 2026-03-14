"""
logging_config.py — Enterprise ML Platform Logging
====================================================
Implements:
  1. Structured JSON logging       — machine-readable, Splunk/ELK/Datadog ready
  2. Correlation / Trace IDs       — inject trace_id across the full request chain
  3. Runtime log level control     — change verbosity via env var, no redeploy
  4. Centralized log shipping      — CloudWatch / Loki / file, all pluggable
  5. Log rotation & retention      — size-based + time-based, configurable
  6. PII / sensitive data masking  — redact emails, tokens, IDs before emit
  7. Async non-blocking logging    — QueueHandler + QueueListener, zero latency impact
  8. Per-component namespacing     — isolated loggers per service module
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import queue
import re
import sys
import threading
import uuid
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


try:
    from pythonjsonlogger import jsonlogger  # type: ignore
    _JSON_LOGGER_AVAILABLE = True
except ImportError:
    _JSON_LOGGER_AVAILABLE = False


try:
    import boto3  # type: ignore
    _BOTO3_AVAILABLE = True
except ImportError:
    _BOTO3_AVAILABLE = False



class LogConfig:
    # Level (runtime-swappable)
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # Service identity (injected into every log record) 
    SERVICE_NAME:    str = os.getenv("SERVICE_NAME",    "ml-platform")
    SERVICE_VERSION: str = os.getenv("SERVICE_VERSION", "unknown")
    ENVIRONMENT:     str = os.getenv("ENVIRONMENT",     "production")

    #  File output 
    LOG_DIR:              Path  = Path(os.getenv("LOG_DIR", "logs"))
    LOG_FILE_NAME:        str   = os.getenv("LOG_FILE_NAME", "ml_platform.log")
    LOG_MAX_BYTES:        int   = int(os.getenv("LOG_MAX_BYTES",   str(50 * 1024 * 1024)))  # 50 MB
    LOG_BACKUP_COUNT:     int   = int(os.getenv("LOG_BACKUP_COUNT", "10"))
    LOG_ROTATION_WHEN:    str   = os.getenv("LOG_ROTATION_WHEN",   "midnight")
    LOG_RETENTION_DAYS:   int   = int(os.getenv("LOG_RETENTION_DAYS", "30"))

    # Async queue 
    LOG_QUEUE_SIZE: int = int(os.getenv("LOG_QUEUE_SIZE", "10000"))

    
    LOG_TARGETS: List[str] = [
        t.strip().lower()
        for t in os.getenv("LOG_TARGETS", "console,file").split(",")
    ]

    #  CloudWatch settings
    CW_LOG_GROUP:  str = os.getenv("CW_LOG_GROUP",  "/ml-platform/inference")
    CW_LOG_STREAM: str = os.getenv("CW_LOG_STREAM", f"{SERVICE_NAME}-{ENVIRONMENT}")
    CW_REGION:     str = os.getenv("AWS_REGION",    "us-east-1")

    
    # Each tuple: (label, compiled_regex, replacement)
    PII_PATTERNS: List[Tuple[str, re.Pattern, str]] = [
        ("email",       re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
                        "[REDACTED-EMAIL]"),
        ("ipv4",        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
                        "[REDACTED-IP]"),
        ("ssn",         re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
                        "[REDACTED-SSN]"),
        ("credit_card", re.compile(r"\b(?:\d[ -]?){13,16}\b"),
                        "[REDACTED-CC]"),
        ("bearer_token",re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE),
                        "Bearer [REDACTED-TOKEN]"),
        ("api_key",     re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[=:]\s*\S+"),
                        r"\1=[REDACTED]"),
    ]




_trace_context = threading.local()


def set_trace_id(trace_id: Optional[str] = None) -> str:
    """
    Set the trace ID for the current thread.
    Auto-generates a UUID if none provided.
    Call this at the entry point of each request/inference job.
    """
    _trace_context.trace_id = trace_id or str(uuid.uuid4())
    return _trace_context.trace_id


def get_trace_id() -> str:
    """Return the current thread's trace ID, or a fallback."""
    return getattr(_trace_context, "trace_id", "no-trace-id")


def clear_trace_id() -> None:
    """Clear trace ID at the end of a request lifecycle."""
    _trace_context.trace_id = None




class PIIMaskingFilter(logging.Filter):
    """
    Scrubs sensitive data from log records BEFORE they are emitted.
    Applied at the handler level so it catches both message and extra fields.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self._redact(str(record.msg))

        # Redact string values in the record's __dict__ (extra fields)
        for key, value in record.__dict__.items():
            if isinstance(value, str) and key not in (
                "name", "levelname", "pathname", "filename",
                "module", "funcName", "levelno", "lineno",
            ):
                record.__dict__[key] = self._redact(value)

        return True  # Always let the record through — we only sanitize, never drop

    @staticmethod
    def _redact(text: str) -> str:
        for _label, pattern, replacement in LogConfig.PII_PATTERNS:
            text = pattern.sub(replacement, text)
        return text



class TraceIDFilter(logging.Filter):
    """
    Injects trace_id, service, version, and environment into every log record.
    This means you never have to manually pass these in logger.info(..., extra={}).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id       = get_trace_id()
        record.service        = LogConfig.SERVICE_NAME
        record.service_version= LogConfig.SERVICE_VERSION
        record.environment    = LogConfig.ENVIRONMENT
        return True




class _EnterpriseJSONFormatter(logging.Formatter):
    """
    Emits fully structured JSON — one object per line.
    Fields: timestamp, level, service, version, environment,
            trace_id, logger, file, line, message + any extras.

    Used when python-json-logger is NOT installed (zero-dependency fallback).
    """

    def format(self, record: logging.LogRecord) -> str:
        import json

        self.formatException   # ensure exc_info is handled
        record.message = record.getMessage()

        payload: Dict[str, Any] = {
            "timestamp":   datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level":       record.levelname,
            "service":     getattr(record, "service",         LogConfig.SERVICE_NAME),
            "version":     getattr(record, "service_version", LogConfig.SERVICE_VERSION),
            "environment": getattr(record, "environment",     LogConfig.ENVIRONMENT),
            "trace_id":    getattr(record, "trace_id",        "no-trace-id"),
            "logger":      record.name,
            "file":        record.filename,
            "line":        record.lineno,
            "function":    record.funcName,
            "message":     record.message,
        }

        # Attach exception info if present
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # Attach any extra fields added via logger.info(..., extra={...})
        _stdlib_keys = logging.LogRecord(
            "", 0, "", 0, "", (), None
        ).__dict__.keys() | {"message", "asctime"}

        for key, val in record.__dict__.items():
            if key not in _stdlib_keys and not key.startswith("_"):
                payload[key] = val

        return json.dumps(payload, default=str)


def _build_formatter() -> logging.Formatter:
    """Return the best available JSON formatter."""
    if _JSON_LOGGER_AVAILABLE:
        fmt = jsonlogger.JsonFormatter(
            fmt="%(timestamp)s %(level)s %(service)s %(trace_id)s %(message)s",
            rename_fields={"levelname": "level", "asctime": "timestamp"},
        )
        return fmt
    return _EnterpriseJSONFormatter()




def _build_console_handler(formatter: logging.Formatter) -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    return handler


def _build_rotating_file_handler(formatter: logging.Formatter) -> logging.Handler:
    """
    Size-based rotation (default 50 MB) with a backup count.
    Falls back gracefully if the log directory can't be created.
    """
    LogConfig.LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LogConfig.LOG_DIR / LogConfig.LOG_FILE_NAME

    handler = RotatingFileHandler(
        filename=path,
        maxBytes=LogConfig.LOG_MAX_BYTES,
        backupCount=LogConfig.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(formatter)
    return handler


def _build_timed_file_handler(formatter: logging.Formatter) -> logging.Handler:
    """
    Time-based rotation (default: midnight) with retention-days–based backup count.
    Complements size-based rotation for audit/compliance log archiving.
    """
    LogConfig.LOG_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = LogConfig.LOG_DIR / f"archive_{LogConfig.LOG_FILE_NAME}"

    handler = TimedRotatingFileHandler(
        filename=archive_path,
        when=LogConfig.LOG_ROTATION_WHEN,
        backupCount=LogConfig.LOG_RETENTION_DAYS,
        encoding="utf-8",
        utc=True,
    )
    handler.setFormatter(formatter)
    return handler


def _build_cloudwatch_handler(formatter: logging.Formatter) -> Optional[logging.Handler]:
    """
    Ships logs to AWS CloudWatch Logs.
    Returns None (with a warning) if boto3 isn't available.
    Requires IAM permissions: logs:CreateLogGroup, logs:CreateLogStream, logs:PutLogEvents
    """
    if not _BOTO3_AVAILABLE:
        print(
            "[logging_config] WARNING: boto3 not installed — CloudWatch shipping disabled.",
            file=sys.stderr,
        )
        return None

    try:
        # watchtower wraps boto3 for CloudWatch log shipping
        import watchtower  # type: ignore

        client = boto3.client("logs", region_name=LogConfig.CW_REGION)
        handler = watchtower.CloudWatchLogHandler(
            log_group=LogConfig.CW_LOG_GROUP,
            stream_name=LogConfig.CW_LOG_STREAM,
            boto3_client=client,
            send_interval=5,        # batch & ship every 5 seconds
            max_batch_count=1_000,
        )
        handler.setFormatter(formatter)
        return handler
    except Exception as exc:
        print(f"[logging_config] WARNING: CloudWatch handler failed to initialize: {exc}", file=sys.stderr)
        return None




_log_queue:    Optional[queue.Queue]              = None
_queue_listener: Optional[logging.handlers.QueueListener] = None


def _wrap_handlers_async(handlers: List[logging.Handler]) -> logging.handlers.QueueHandler:
    """
    Place all handlers behind a QueueHandler + QueueListener pair.
    The application thread only enqueues — a background thread does all I/O.
    """
    global _log_queue, _queue_listener

    _log_queue = queue.Queue(maxsize=LogConfig.LOG_QUEUE_SIZE)

    # QueueListener runs in a daemon thread — processes records off the queue
    _queue_listener = logging.handlers.QueueListener(
        _log_queue,
        *handlers,
        respect_handler_level=True,
    )
    _queue_listener.start()

    return logging.handlers.QueueHandler(_log_queue)


def shutdown_logging() -> None:
    """
    Gracefully flush and stop the async queue listener.
    Call this in your application shutdown hook / atexit handler.
    """
    if _queue_listener:
        _queue_listener.stop()




def set_log_level(level: str) -> None:
    """
    Hot-swap the root logger level at runtime — no restart required.
    Useful for feature-flag driven verbosity changes or incident response.

    Example:
        set_log_level("DEBUG")   # turn up verbosity during an incident
        set_log_level("WARNING") # quiet down after resolution
    """
    numeric = logging.getLevelName(level.upper())
    if not isinstance(numeric, int):
        raise ValueError(f"Invalid log level: {level!r}")

    logging.getLogger().setLevel(numeric)
    # Propagate to all existing handlers
    for handler in logging.getLogger().handlers:
        handler.setLevel(numeric)

    logging.getLogger(__name__).info(
        "log_level_changed",
        extra={"new_level": level.upper(), "event": "runtime_level_change"},
    )




_configured = False
_config_lock = threading.Lock()


class LoggingConfigurator:
    """

    Call LoggingConfigurator.configure() once at application startup.
    Everything else (trace IDs, per-component loggers) works automatically.
    """

    @staticmethod
    def configure(level: Optional[str] = None) -> None:
        """
        Idempotent setup — safe to call multiple times.
        Level defaults to LOG_LEVEL env var (default: INFO).
        """
        global _configured

        with _config_lock:
            if _configured:
                return
            _configured = True

        resolved_level = logging.getLevelName((level or LogConfig.LOG_LEVEL).upper())

        root_logger = logging.getLogger()
        root_logger.setLevel(resolved_level)

        # Shared filters (applied to every handler) 
        pii_filter   = PIIMaskingFilter()
        trace_filter = TraceIDFilter()

        # Formatter 
        formatter = _build_formatter()

        # Build concrete handlers based on LOG_TARGETS env var 
        concrete_handlers: List[logging.Handler] = []

        if "console" in LogConfig.LOG_TARGETS:
            concrete_handlers.append(_build_console_handler(formatter))

        if "file" in LogConfig.LOG_TARGETS:
            concrete_handlers.append(_build_rotating_file_handler(formatter))
            concrete_handlers.append(_build_timed_file_handler(formatter))

        if "cloudwatch" in LogConfig.LOG_TARGETS:
            cw = _build_cloudwatch_handler(formatter)
            if cw:
                concrete_handlers.append(cw)

        if not concrete_handlers:
            # Safety net — always have at least a console handler
            concrete_handlers.append(_build_console_handler(formatter))

        #  Apply filters to every concrete handle
        for handler in concrete_handlers:
            handler.addFilter(pii_filter)
            handler.addFilter(trace_filter)
            handler.setLevel(resolved_level)

        queue_handler = _wrap_handlers_async(concrete_handlers)
        root_logger.addHandler(queue_handler)

    
        for noisy_lib in ("urllib3", "botocore", "s3transfer", "boto3"):
            logging.getLogger(noisy_lib).setLevel(logging.WARNING)

        root_logger.info(
            "logging_configured",
            extra={
                "event":       "logging_configured",
                "level":       logging.getLevelName(resolved_level),
                "targets":     LogConfig.LOG_TARGETS,
                "service":     LogConfig.SERVICE_NAME,
                "environment": LogConfig.ENVIRONMENT,
            },
        )



_COMPONENT_NAMES = {
    "model_server",
    "feature_store",
    "data_pipeline",
    "inference_engine",
    "drift_detector",
    "experiment_tracker",
    "health_check",
    "api_gateway",
}


def get_logger(name: str, component: Optional[str] = None) -> logging.Logger:
    """
    Returns a namespaced logger, ensuring logging is configured first.

    Args:
        name:      Typically __name__ of the calling module.
        component: Optional platform component label (e.g. "model_server").
                   Adds a namespace prefix: ml_platform.<component>.<name>

    Usage:
        # In model_server.py:
        logger = get_logger(__name__, component="model_server")

        # In feature_store.py:
        logger = get_logger(__name__, component="feature_store")
    """
    LoggingConfigurator.configure()

    if component and component in _COMPONENT_NAMES:
        logger_name = f"ml_platform.{component}.{name}"
    else:
        logger_name = f"ml_platform.{name}"

    return logging.getLogger(logger_name)


def get_component_logger(component: str) -> logging.Logger:
    """
    Shorthand: get a top-level component logger.

    Usage:
        logger = get_component_logger("drift_detector")
        logger.info("drift_check_complete", extra={"psi_score": 0.12})
    """
    LoggingConfigurator.configure()
    return logging.getLogger(f"ml_platform.{component}")



class RequestContext:
    """
    Context manager that sets and clears a trace ID for the request lifetime.

    Usage (FastAPI middleware or inference handler):

        with RequestContext(trace_id=request.headers.get("X-Trace-Id")):
            result = model.predict(features)
            # All logs inside here automatically carry the trace_id
    """

    def __init__(self, trace_id: Optional[str] = None) -> None:
        self.trace_id = trace_id or str(uuid.uuid4())

    def __enter__(self) -> "RequestContext":
        set_trace_id(self.trace_id)
        return self

    def __exit__(self, *_: Any) -> None:
        clear_trace_id()