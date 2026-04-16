from __future__ import annotations

import polars as pl
import numpy as np
from typing import List, Optional, Literal
from dataclasses import dataclass, field



# Structured Report


@dataclass
class DataQualityReport:
    status: Literal["valid", "warning", "invalid"]
    row_count: int
    column_count: int
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    quality_score: float = 1.0



# Data Quality Analyzer


class DataQualityAnalyzer:
    """
    Production-grade Data Quality Firewall.

    This class performs:
    - Structural validation
    - Statistical anomaly detection
    - Behavioral risk checks

    It NEVER mutates input data.
    """

    NULL_THRESHOLD = 0.6
    CONSTANT_THRESHOLD = 0.8
    MIN_ROWS_REQUIRED = 10
    IMBALANCE_THRESHOLD = 0.95

    def __init__(self, df: pl.DataFrame, target_column: Optional[str] = None):
        self.df = df
        self.target_column = target_column

        self.issues: List[str] = []
        self.warnings: List[str] = []
        self.recommendations: List[str] = []

    
    # Public API
    

    def analyze(self) -> DataQualityReport:
        self._check_empty_dataset()
        self._check_all_null_rows()
        self._check_zero_variance_columns()
        self._check_duplicate_rows()

        self._check_high_null_columns()
        self._check_mixed_types()
        self._check_numeric_as_string()
        self._check_malformed_datetime()

        self._check_high_constant_columns()
        self._check_target_leakage()
        self._check_target_imbalance()

        quality_score = self._compute_quality_score()
        status = self._determine_status(quality_score)

        return DataQualityReport(
            status=status,
            row_count=self.df.height,
            column_count=self.df.width,
            issues=self.issues,
            warnings=self.warnings,
            recommendations=self.recommendations,
            quality_score=quality_score,
        )

    
    # Structural Failures
    

    def _check_empty_dataset(self):
        if self.df.height == 0:
            self.issues.append("Dataset is empty.")
            self.recommendations.append("Upload a dataset with at least one row.")

        if self.df.height < self.MIN_ROWS_REQUIRED:
            self.warnings.append(
                f"Dataset contains fewer than {self.MIN_ROWS_REQUIRED} rows."
            )

    
    def _check_all_null_rows(self):
        if self.df.height == 0:
            return

        null_rows = (
            self.df
            .select(pl.all().is_null().all())
            .to_series()
            .sum()
        )

        if null_rows > 0:
            self.issues.append("Dataset contains rows where all values are null.")
            self.recommendations.append("Remove fully null rows.")


    def _check_zero_variance_columns(self):
        for col in self.df.columns:
            if self.df[col].n_unique() <= 1:
                self.warnings.append(f"Column '{col}' has zero variance.")

    def _check_duplicate_rows(self):
        duplicate_count = self.df.height - self.df.unique().height
        if duplicate_count > 0:
            self.warnings.append(f"Dataset contains {duplicate_count} duplicate rows.")
            self.recommendations.append("Consider removing duplicate rows.")

    
    # Statistical Risk

    def _check_high_null_columns(self):
        for col in self.df.columns:
            null_ratio = self.df[col].null_count() / max(self.df.height, 1)
            if null_ratio > self.NULL_THRESHOLD:
                self.warnings.append(
                    f"Column '{col}' has {null_ratio:.0%} null values."
                )

    def _check_mixed_types(self):
        for col in self.df.columns:
            try:
                self.df[col].cast(pl.Float64)
            except Exception:
                if self.df[col].dtype == pl.Object:
                    self.warnings.append(
                        f"Column '{col}' may contain mixed data types."
                    )

    def _check_numeric_as_string(self):
        for col in self.df.columns:
            if self.df[col].dtype == pl.Utf8:
                try:
                    self.df[col].cast(pl.Float64)
                    self.warnings.append(
                        f"Column '{col}' appears numeric but stored as string."
                    )
                except Exception:
                    pass

    def _check_malformed_datetime(self):
        for col in self.df.columns:
            if self.df[col].dtype == pl.Utf8:
                try:
                    self.df[col].str.strptime(pl.Datetime, strict=True)
                except Exception:
                    continue
                else:
                    self.warnings.append(
                        f"Column '{col}' may represent datetime but needs explicit parsing."
                    )

    
    # Behavioral Risk

    def _check_high_constant_columns(self):
        for col in self.df.columns:
            if self.df.height == 0:
                continue

            vc = self.df[col].value_counts()

            if vc.height == 0:
                continue

            most_freq = (
                vc.sort("count", descending=True)
                .select("count")
                .to_series()
                .item(0)
            )

            ratio = most_freq / self.df.height

            if ratio > self.CONSTANT_THRESHOLD:
                self.warnings.append(
                    f"Column '{col}' is {ratio:.0%} constant."
                )


    def _check_target_leakage(self):
        if not self.target_column:
            return

        for col in self.df.columns:
            if col == self.target_column:
                continue

            if self.df[col].equals(self.df[self.target_column]):
                self.issues.append(
                    f"Column '{col}' is identical to target column (leakage)."
                )

    def _check_target_imbalance(self):
        if not self.target_column:
            return

        if self.target_column not in self.df.columns:
            return

        target = self.df[self.target_column]

        if target.dtype not in (pl.Int64, pl.Utf8):
            return

        if self.df.height == 0:
            return

        vc = target.value_counts()

        if vc.height == 0:
            return

        distribution = (
            vc.sort("count", descending=True)
            .select("count")
            .to_series()
        )

        if len(distribution) > 0:
            ratio = distribution.item(0) / self.df.height

            if ratio > self.IMBALANCE_THRESHOLD:
                self.warnings.append(
                    f"Target column '{self.target_column}' is extremely imbalanced ({ratio:.0%} dominant class)."
                )
                self.recommendations.append(
                    "Consider resampling or class weighting."
                )


        # Scoring Logic
    

    def _compute_quality_score(self) -> float:
        penalty = len(self.issues) * 0.15 + len(self.warnings) * 0.05
        score = max(0.0, 1.0 - penalty)
        return round(score, 3)

    def _determine_status(self, score: float) -> Literal["valid", "warning", "invalid"]:
        if self.issues:
            return "invalid"
        if score < 0.7:
            return "warning"
        return "valid"
