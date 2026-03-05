import numpy as np
import polars as pl

from core.drift.drift_detector import DriftDetector


def test_no_drift_detected():
    np.random.seed(42)

    reference = pl.DataFrame({
        "feature_1": np.random.normal(0, 1, 1000),
        "feature_2": np.random.normal(5, 2, 1000),
    })

    # Same distribution
    current = pl.DataFrame({
        "feature_1": np.random.normal(0, 1, 1000),
        "feature_2": np.random.normal(5, 2, 1000),
    })

    detector = DriftDetector(threshold=0.01)
    report = detector.check_drift(reference, current)

    assert report.is_drifted is False
    assert len(report.flagged_features) == 0


def test_drift_detected():
    np.random.seed(42)

    reference = pl.DataFrame({
        "feature_1": np.random.normal(0, 1, 1000),
    })

    # Different distribution
    current = pl.DataFrame({
        "feature_1": np.random.normal(5, 1, 1000),
    })

    detector = DriftDetector(threshold=0.05)
    report = detector.check_drift(reference, current)

    assert report.is_drifted is True
    assert "feature_1" in report.flagged_features


def test_non_numeric_column_ignored():
    reference = pl.DataFrame({
        "feature_1": [1, 2, 3, 4],
        "category": ["A", "B", "A", "C"],
    })

    current = pl.DataFrame({
        "feature_1": [1, 2, 3, 4],
        "category": ["A", "A", "B", "C"],
    })

    detector = DriftDetector()
    report = detector.check_drift(reference, current)

    # category should be ignored
    assert "category" not in report.drift_scores


def test_missing_column_handled():
    reference = pl.DataFrame({
        "feature_1": [1, 2, 3, 4],
        "feature_2": [10, 20, 30, 40],
    })

    current = pl.DataFrame({
        "feature_1": [1, 2, 3, 4],
        # feature_2 missing
    })

    detector = DriftDetector()
    report = detector.check_drift(reference, current)

    # Should not crash
    assert "feature_2" not in report.drift_scores