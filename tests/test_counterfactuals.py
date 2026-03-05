import pytest
import polars as pl
import numpy as np

from core.decision_engine.counterfactuals import (
    CounterfactualOrchestrator,
    OptimizationStatus,
)



# Dummy Model


class DummyModel:
    def predict(self, df: pl.DataFrame):
        # Simple linear behavior for test stability
        return np.array([df.select(pl.all()).to_numpy().sum()])



# Dummy Optimizer Patch


class DummyOptimizer:
    def __init__(self, model):
        self.model = model

    def optimize(self, row, lever_col, target_goal, bounds):
        # Always return midpoint for deterministic behavior
        return (bounds[0] + bounds[1]) / 2



# Fixture


@pytest.fixture
def sample_training_data():
    return pl.DataFrame({
        "feature1": np.random.normal(10, 2, 100),
        "feature2": np.random.normal(5, 1, 100),
    })


@pytest.fixture
def single_row():
    return pl.DataFrame({
        "feature1": [10.0],
        "feature2": [5.0],
    })



# Tests


def test_success_safe_mode(sample_training_data, single_row, monkeypatch):

    model = DummyModel()
    orchestrator = CounterfactualOrchestrator(
        model=model,
        training_data=sample_training_data,
    )

    # Patch optimizer for deterministic behavior
    monkeypatch.setattr(
        orchestrator,
        "_optimizer",
        DummyOptimizer(model)
    )

    result = orchestrator.explain_how_to_hit_target(
        original_row=single_row,
        target_goal=20.0,
        lever_col="feature1",
        bounds=(5.0, 15.0),
        strict=False,
    )

    assert result.status in {
        OptimizationStatus.SAFE,
        OptimizationStatus.MODERATE_RISK,
        OptimizationStatus.HIGH_RISK,
        OptimizationStatus.UNREACHABLE,
    }

    assert 0.0 <= result.risk_score <= 1.0


def test_strict_mode_raises(sample_training_data, single_row):

    model = DummyModel()
    orchestrator = CounterfactualOrchestrator(
        model=model,
        training_data=sample_training_data,
    )

    # Invalid bounds
    with pytest.raises(ValueError):
        orchestrator.explain_how_to_hit_target(
            original_row=single_row,
            target_goal=10.0,
            lever_col="feature1",
            bounds=(10.0, 5.0),
            strict=True,
        )


def test_safe_mode_returns_failed(sample_training_data, single_row):

    model = DummyModel()
    orchestrator = CounterfactualOrchestrator(
        model=model,
        training_data=sample_training_data,
    )

    # Invalid bounds but safe mode
    result = orchestrator.explain_how_to_hit_target(
        original_row=single_row,
        target_goal=10.0,
        lever_col="feature1",
        bounds=(10.0, 5.0),
        strict=False,
    )

    assert result.status == OptimizationStatus.FAILED
    assert result.risk_score == 1.0


def test_missing_feature_strict(sample_training_data, single_row):

    model = DummyModel()
    orchestrator = CounterfactualOrchestrator(
        model=model,
        training_data=sample_training_data,
    )

    with pytest.raises(KeyError):
        orchestrator.explain_how_to_hit_target(
            original_row=single_row,
            target_goal=10.0,
            lever_col="non_existing_feature",
            bounds=(0.0, 10.0),
            strict=True,
        )


def test_non_numeric_feature(sample_training_data):

    model = DummyModel()

    row = pl.DataFrame({
        "feature1": [10.0],
        "feature2": ["not_numeric"],
    })

    orchestrator = CounterfactualOrchestrator(
        model=model,
        training_data=sample_training_data,
    )

    with pytest.raises(TypeError):
        orchestrator.explain_how_to_hit_target(
            original_row=row,
            target_goal=10.0,
            lever_col="feature2",
            bounds=(0.0, 10.0),
            strict=True,
        )


def test_unreachable_status(sample_training_data, single_row, monkeypatch):

    model = DummyModel()
    orchestrator = CounterfactualOrchestrator(
        model=model,
        training_data=sample_training_data,
        reachability_tolerance=0.0001,
    )

    monkeypatch.setattr(
        orchestrator,
        "_optimizer",
        DummyOptimizer(model)
    )

    result = orchestrator.explain_how_to_hit_target(
        original_row=single_row,
        target_goal=1000000.0,  # Impossible target
        lever_col="feature1",
        bounds=(5.0, 15.0),
        strict=False,
    )

    assert result.status == OptimizationStatus.UNREACHABLE
