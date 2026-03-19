from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import polars as pl

# Configure Structured Logging
logger = logging.getLogger(__name__)

class ImputationStrategy(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"

@dataclass(frozen=True)
class PreprocessingMetadata:
    """Audit trail for data transformations."""
    dropped_features: List[str] = field(default_factory=list)
    leaked_features: List[str] = field(default_factory=list)
    duplicates_removed: int = 0
    is_imbalanced: bool = False
    execution_time_ms: float = 0.0
    

class FeatureProcessor:
    """
    High-Performance Stateless Feature Engineering Engine.
    Optimized for Polars LazyFrames to ensure minimal memory overhead.
    """
    
    def __init__(
        self,
        target_column: str,
        numeric_strategy: ImputationStrategy = ImputationStrategy.MEDIAN,
        categorical_strategy: ImputationStrategy = ImputationStrategy.MODE,
        variance_threshold: int = 1,
        leakage_threshold: float = 0.995
        
        
    ):
        self.target_column = target_column
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.variance_threshold = variance_threshold
        self.leakage_threshold = leakage_threshold
        
        

    def process(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, PreprocessingMetadata]:
        
        
        """
        Main entry point. Orchestrates the cleaning pipeline.
        Uses a 'Fail-Fast' approach for data integrity.
        """
        if df.height == 0:
            raise ValueError("Inference failed: DataFrame contains zero records.")

        #  Deduplication (Eager - changes row count)
        df, removed_count = self._remove_duplicates(df)

        #  Type Discovery
        schema = df.schema
        numeric_cols = [c for c, t in schema.items() if t.is_numeric() and c != self.target_column]
        cat_cols = [c for c, t in schema.items() if (t.is_temporal() or t == pl.Utf8 or t == pl.Categorical) and c != self.target_column]

        #  Leakage & Variance Detection
        leaked = self._detect_leakage(df, numeric_cols)
        low_variance = self._detect_low_variance(df, numeric_cols + cat_cols)
        to_drop = list(set(leaked + low_variance))

        # Final Pipeline Construction (Lazy)
        df_lazy = df.lazy().drop(to_drop)
        
        # Apply vectorised imputation and encoding in a single graph
        df_lazy = self._apply_imputation(df_lazy, numeric_cols, cat_cols, to_drop)
        df_lazy = self._apply_encoding(df_lazy, cat_cols, to_drop)

        #  Imbalance Check
        is_imbalanced = self._check_imbalance(df)

        return df_lazy.collect(), PreprocessingMetadata(
            dropped_features=low_variance,
            leaked_features=leaked,
            duplicates_removed=removed_count,
            is_imbalanced=is_imbalanced
        )

    def _remove_duplicates(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, int]:
        clean_df = df.unique(maintain_order=True)
        return clean_df, (df.height - clean_df.height)

    def _detect_leakage(self, df: pl.DataFrame, cols: List[str]) -> List[str]:
        """Mathematically identifies features that 'know too much'."""
        if self.target_column not in df.columns or not df.schema[self.target_column].is_numeric():
            return []

        leaked = []
        for col in cols:
            # Avoid divide by zero/null corr
            corr = df.select(pl.corr(col, self.target_column)).item()
            if corr is not None and abs(corr) >= self.leakage_threshold:
                leaked.append(col)
        return leaked

    def _detect_low_variance(self, df: pl.DataFrame, cols: List[str]) -> List[str]:
        """Identify columns with near-zero information entropy."""
        return [col for col in cols if df[col].n_unique() <= self.variance_threshold]

    def _apply_imputation(
        self, 
        lf: pl.LazyFrame, 
        numeric: List[str], 
        cat: List[str], 
        dropped: List[str]
    ) -> pl.LazyFrame:
        exprs = []
        
        # Numeric
        for col in [c for c in numeric if c not in dropped]:
            if self.numeric_strategy == ImputationStrategy.MEDIAN:
                exprs.append(pl.col(col).fill_null(pl.col(col).median()))
            else:
                exprs.append(pl.col(col).fill_null(pl.col(col).mean()))

        # Categorical
        for col in [c for c in cat if c not in dropped]:
            # .mode().first() ensures we get a single value for the fill
            exprs.append(pl.col(col).fill_null(pl.col(col).mode().first()))

        return lf.with_columns(exprs)

    def _apply_encoding(self, lf: pl.LazyFrame, cat_cols: List[str], dropped: List[str]) -> pl.LazyFrame:
        """Stable physical encoding for categorical data."""
        active_cats = [c for c in cat_cols if c not in dropped]
        return lf.with_columns([
            pl.col(c).cast(pl.Categorical).to_physical() for c in active_cats
        ])

    def _check_imbalance(self, df: pl.DataFrame) -> bool:
        if self.target_column not in df.columns: 
            return False
            
        # value_counts() in Polars returns [target_column_name, "count"]
        counts_df = df[self.target_column].value_counts()
        
        if counts_df.height < 2: 
            return False
            
        # Dynamically find the count column (usually "count")
        count_col = "count" if "count" in counts_df.columns else counts_df.columns[1]
        
        return (counts_df[count_col].max() / df.height) >= 0.80
    
    