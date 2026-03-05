from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class CircuitBreakerTriggered(Exception):
    """Raised when execution must be halted immediately."""
    pass


@dataclass
class CircuitBreakerConfig:
    min_quality_score: float = 0.4
    max_drift_score: float = 0.3
    min_model_performance: float = 0.5
    max_training_time_seconds: int = 600
    max_failure_count: int = 5


class CircuitBreaker:
    """
    Enterprise-grade ML execution guard.

    Stops execution when:
    - Data quality too low
    - Drift too high
    - Model performance unacceptable
    - Training time exceeded
    - System instability detected
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()

    # Data Quality Gate

    def check_data_quality(self, quality_score: float):
        if quality_score < self.config.min_quality_score:
            raise CircuitBreakerTriggered(
                f"Data quality too low ({quality_score}). Execution halted."
            )


    # Drift Gate
    
    def check_drift(self, drift_score: float):
        if drift_score > self.config.max_drift_score:
            raise CircuitBreakerTriggered(
                f"Data drift too high ({drift_score}). Retraining required."
            )
    
    # Model Performance Gate
    
    def check_model_performance(self, metric_value: float):
        if metric_value < self.config.min_model_performance:
            raise CircuitBreakerTriggered(
                f"Model performance below threshold ({metric_value})."
            )
    
    # Training Time Gate
    
    def check_training_time(self, elapsed_seconds: float):
        if elapsed_seconds > self.config.max_training_time_seconds:
            raise CircuitBreakerTriggered(
                f"Training exceeded time limit ({elapsed_seconds}s)."
            )

    # Failure Gate
    
    def check_failure_count(self, failure_count: int):
        if failure_count >= self.config.max_failure_count:
            raise CircuitBreakerTriggered(
                f"Too many consecutive failures ({failure_count})."
            )
