import polars as pl
import numpy as np
from typing import List, Optional
from enum import Enum
from dataclasses import dataclass

class TemporalResolution(str, Enum):
    HIGH = "high"    # Hour, Minute, Second
    LOW = "low"      # Year, Month, Day, Weekday
    CYCLIC = "cyclic" # Sine/Cosine transformations

@dataclass(frozen=True)
class TemporalFeatureConfig:
    """Configuration for temporal signal extraction."""
    resolution: TemporalResolution = TemporalResolution.LOW
    add_is_weekend: bool = True
    add_cyclic_signals: bool = True

class TemporalFeatureEngineer:
    """
    Transforms raw Timestamps into high-dimensional feature vectors.
    """

    _TEMPORAL_DTYPES = (pl.Date, pl.Datetime)

    def __init__(self, config: Optional[TemporalFeatureConfig] = None):
        self.config = config or TemporalFeatureConfig()

    def transform(self, df: pl.DataFrame, date_cols: List[str]) -> pl.DataFrame:
        """
        Applies a vectorized transformation pipeline to specified datetime columns.
        Formula Name: Temporal Signal Decomposition
        """
        
        missing = [c for c in date_cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"TemporalFeatureEngineer.transform: column(s) {missing} not found "
                f"in dataframe. Available columns: {df.columns}"
            )

        non_temporal = [
            c for c in date_cols if df.schema[c] not in self._TEMPORAL_DTYPES
        ]
        if non_temporal:
            raise TypeError(
                f"TemporalFeatureEngineer.transform: column(s) {non_temporal} are not "
                f"Date/Datetime dtype (got {[str(df.schema[c]) for c in non_temporal]}). "
                "Parse these columns to a temporal dtype before calling transform()."
            )

        expressions = []

        for col in date_cols:
            expressions.extend([
                pl.col(col).dt.year().alias(f"{col}_year"),
                pl.col(col).dt.month().alias(f"{col}_month"),
                pl.col(col).dt.day().alias(f"{col}_day"),
                pl.col(col).dt.weekday().alias(f"{col}_weekday"),
            ])

            if self.config.resolution == TemporalResolution.HIGH:
                expressions.extend([
                    pl.col(col).dt.hour().alias(f"{col}_hour"),
                    pl.col(col).dt.minute().alias(f"{col}_minute"),
                ])

           
            if self.config.add_is_weekend:
                expressions.append(
                    (pl.col(col).dt.weekday() >= 6).alias(f"{col}_is_weekend")
                )

            # Cyclic Encoding (Trigonometric Features)
            if self.config.add_cyclic_signals:
                expressions.extend(self._get_cyclic_exprs(col))

        return df.with_columns(expressions)

    def _get_cyclic_exprs(self, col: str) -> List[pl.Expr]:
        """
        Encodes time as a circle to maintain distance integrity (e.g., Dec to Jan).
        Formula: x_sin = sin(2 * pi * x / max_x)
        """
        return [
            # Monthly Cycle
            (pl.col(col).dt.month() * (2 * np.pi / 12)).sin().alias(f"{col}_month_sin"),
            (pl.col(col).dt.month() * (2 * np.pi / 12)).cos().alias(f"{col}_month_cos"),

            # Weekday Cycle
            (pl.col(col).dt.weekday() * (2 * np.pi / 7)).sin().alias(f"{col}_weekday_sin"),
            (pl.col(col).dt.weekday() * (2 * np.pi / 7)).cos().alias(f"{col}_weekday_cos"),
        ]