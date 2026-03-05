import pytest
from core.contracts.circuit_breaker import CircuitBreaker, CircuitBreakerTriggered


def test_quality_triggers():
    breaker = CircuitBreaker()

    with pytest.raises(CircuitBreakerTriggered):
        breaker.check_data_quality(0.2)


def test_drift_triggers():
    breaker = CircuitBreaker()

    with pytest.raises(CircuitBreakerTriggered):
        breaker.check_drift(0.5)


def test_model_performance_triggers():
    breaker = CircuitBreaker()

    with pytest.raises(CircuitBreakerTriggered):
        breaker.check_model_performance(0.3)
