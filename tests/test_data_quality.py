import pytest
import polars as pl

from core.feature_engineering.data_quality import DataQualityAnalyzer



# Empty Dataset → INVALID


def test_empty_dataframe_invalid():
    df = pl.DataFrame()

    analyzer = DataQualityAnalyzer(df)
    report = analyzer.analyze()

    assert report.status == "invalid"
    assert "Dataset is empty." in report.issues
    assert report.quality_score < 1.0



# Dataset with Duplicates → WARNING


def test_duplicate_rows_warning():
    df = pl.DataFrame({
        "feature": [1, 2, 2, 3],
        "target": [10, 20, 20, 30],
    })

    analyzer = DataQualityAnalyzer(df, target_column="target")
    report = analyzer.analyze()

    assert report.status in ["valid", "warning"]
    assert any("duplicate rows" in w.lower() for w in report.warnings)



#  Target Leakage → INVALID


def test_target_leakage_invalid():
    df = pl.DataFrame({
        "feature": [1, 2, 3, 4],
        "target": [10, 20, 30, 40],
        "leak_column": [10, 20, 30, 40],  # identical to target
    })

    analyzer = DataQualityAnalyzer(df, target_column="target")
    report = analyzer.analyze()

    assert report.status == "invalid"
    assert any("leakage" in issue.lower() for issue in report.issues)



# Imbalanced Classification → WARNING


def test_imbalanced_target_warning():
    df = pl.DataFrame({
        "feature": list(range(100)),
        "target": [1] * 98 + [0] * 2,  # 98% one class
    })

    analyzer = DataQualityAnalyzer(df, target_column="target")
    report = analyzer.analyze()

    assert report.status in ["valid", "warning"]
    assert any("imbalanced" in w.lower() for w in report.warnings)



#  Clean Dataset → VALID


def test_clean_dataset_valid():
    df = pl.DataFrame({
        "feature1": list(range(50)),
        "feature2": list(range(50, 100)),
        "target": [0, 1] * 25,
    })

    analyzer = DataQualityAnalyzer(df, target_column="target")
    report = analyzer.analyze()

    assert report.status == "valid"
    assert len(report.issues) == 0
    assert report.quality_score >= 0.7
