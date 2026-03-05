from __future__ import annotations

import polars as pl
from typing import Dict, List, Optional
from dataclasses import dataclass


# TYPE REGISTRY


NUMERIC_DTYPES = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
}

DATETIME_DTYPES = {pl.Date, pl.Datetime}

CATEGORICAL_DTYPES = {pl.Utf8, pl.Categorical}



# METADATA MODELS

@dataclass
class FeatureMetadata:
    numeric: List[str]
    categorical: List[str]
    datetime: List[str]
    target: Optional[str]
    problem_type: Optional[str]



# TABULAR INTELLIGENCE ENGINE

class TabularIntelligenceEngine:
    """
    Enterprise AutoML intelligence layer.

    Responsibilities:
    - Automatic feature classification
    - Problem type detection
    - Safe feature engineering
    """

    def __init__(self, df: pl.DataFrame, target_column: Optional[str] = None):
        self.df = df
        self.target_column = target_column


    # FEATURE CLASSIFICATION

    def classify_features(self) -> FeatureMetadata:

        numeric = []
        categorical = []
        datetime = []

        for col, dtype in self.df.schema.items():

            if col == self.target_column:
                continue

            if dtype in NUMERIC_DTYPES:
                numeric.append(col)

            elif dtype in DATETIME_DTYPES:
                datetime.append(col)

            elif dtype in CATEGORICAL_DTYPES:
                categorical.append(col)

            else:
                categorical.append(col)

        problem_type = self._detect_problem_type()

        return FeatureMetadata(
            numeric=numeric,
            categorical=categorical,
            datetime=datetime,
            target=self.target_column,
            problem_type=problem_type,
        )


    # PROBLEM TYPE DETECTION
    
    def _detect_problem_type(self) -> Optional[str]:

        if not self.target_column:
            return None

        if self.target_column not in self.df.columns:
            return None

        target_series = self.df[self.target_column]
        dtype = target_series.dtype
        total_rows = self.df.height
        unique_values = target_series.n_unique()

        
        # Forecasting detection
        
        if any(t in DATETIME_DTYPES for t in self.df.dtypes):
            if dtype in NUMERIC_DTYPES:
                return "forecasting"

        
        # Classification detection
        
        integer_types = {
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        }

        if dtype in integer_types:
            # Binary classification
            if unique_values == 2:
                return "classification"

            # Multi-class classification (low cardinality)
            cardinality_ratio = unique_values / total_rows
            if unique_values <= 20 and cardinality_ratio < 0.5:
                return "classification"

        
        # Regression detection
    
        if dtype in {pl.Float32, pl.Float64}:
            return "regression"

        
        return None
    

    # FEATURE ENGINEERING
    

    def engineer_features(self) -> pl.DataFrame:
        """
        Safe, deterministic feature engineering.
        """

        df = self.df.clone()

        metadata = self.classify_features()

        
        # Datetime Feature Expansion
        
        for col in metadata.datetime:

            df = df.with_columns([
                pl.col(col).dt.year().alias(f"{col}_year"),
                pl.col(col).dt.month().alias(f"{col}_month"),
                pl.col(col).dt.day().alias(f"{col}_day"),
                pl.col(col).dt.weekday().alias(f"{col}_weekday"),
            ])


        # Numeric Interaction Features
        numeric_cols = metadata.numeric

        if len(numeric_cols) >= 2:
            base = numeric_cols[0]
            for col in numeric_cols[1:3]:  # limit to avoid explosion
                df = df.with_columns(
                    (pl.col(base) * pl.col(col)).alias(f"{base}_x_{col}")
                )

        
        # Safe Log Transform (positive only)
        for col in numeric_cols:
            if (df[col] > 0).all():
                df = df.with_columns(
                    pl.col(col).log().alias(f"{col}_log")
                )

        return df