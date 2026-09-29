from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import polars as pl

from core.contracts.problem_type import ProblemType

logger = logging.getLogger(__name__)


class ImputationStrategy(str, Enum):
    MEAN   = "mean"
    MEDIAN = "median"
    MODE   = "mode"


_SENTINEL_STRINGS: Set[str] = {
    "unknown", "none", "na", "n/a", "null", "nan",
    "nil", "missing", "undefined", "?", "-", "--",
    "not available", "not applicable", "n.a.", "",
}

_NUMERIC_DTYPES  = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)
_TEMPORAL_DTYPES = (pl.Date, pl.Datetime, pl.Duration, pl.Time)
_STRING_DTYPES   = (pl.Utf8, pl.Categorical, pl.Enum)


@dataclass(frozen=True)
class PreprocessingMetadata:
    dropped_features:    List[str] = field(default_factory=list)
    imputed_features:    List[str] = field(default_factory=list)
    target_rows_dropped: int       = 0
    execution_time_ms:   float     = 0.0


class FeatureProcessor:
    def __init__(
        self,
        target_column:        str,
        problem_type:         Optional[ProblemType]  = None,
        numeric_strategy:     ImputationStrategy     = ImputationStrategy.MEDIAN,
        categorical_strategy: ImputationStrategy     = ImputationStrategy.MODE,
        variance_threshold:   int                    = 1,
        null_threshold:       float                  = 0.60,
        protected_columns:    Optional[list[str]]    = None,
        enable_imputation:    bool                   = True,
    ) -> None:
        self.target_column        = target_column
        self.problem_type         = problem_type
        self.numeric_strategy     = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.variance_threshold   = variance_threshold
        self.null_threshold       = null_threshold
        self.protected_columns    = set(protected_columns or [])
        self.enable_imputation    = enable_imputation

        self._fitted         = False
        self._to_drop:       List[str]         = []
        self._numeric_cols:  List[str]         = []
        self._cat_cols:      List[str]         = []
        self._temporal_cols: List[str]         = []
        self._numeric_fills:  Dict[str, float] = {}
        self._category_fills: Dict[str, Any]   = {}
        self._imputed_cols:  List[str]         = []

    def fit_transform(
        self, df: pl.DataFrame
    ) -> Tuple[pl.DataFrame, PreprocessingMetadata]:
        if self._fitted:
            raise RuntimeError(
                "FeatureProcessor is already fitted. "
                "Create a new instance for each training run."
            )
        start = time.perf_counter()

        target_rows_dropped = 0
        if self.target_column in df.columns:
            before = df.height
            df = self._validate_target(df)
            target_rows_dropped = before - df.height

        df = self._clean_feature_strings(df)
        self._classify_columns(df)

        low_var, high_null = self._select_features(df)
        self._to_drop = list(dict.fromkeys(low_var + high_null))

        active_numeric  = [c for c in self._numeric_cols  if c not in self._to_drop]
        active_cat      = [c for c in self._cat_cols      if c not in self._to_drop]
        active_temporal = [c for c in self._temporal_cols if c not in self._to_drop]

        lf = df.lazy().drop(self._to_drop)

        if self.enable_imputation:
            lf = self._fit_imputer(df, lf, active_numeric, active_cat)

        try:
            result_df = lf.collect()
        except Exception:
            logger.exception("fit_transform collect() failed")
            raise

        self._numeric_cols  = active_numeric
        self._cat_cols      = active_cat
        self._temporal_cols = active_temporal
        self._imputed_cols  = list(self._numeric_fills.keys()) + list(self._category_fills.keys())
        self._fitted = True

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "fit_transform complete — %d rows, %d numeric, %d cat, %d dropped, %.1f ms",
            result_df.height, len(active_numeric), len(active_cat),
            len(self._to_drop), elapsed_ms,
        )

        return result_df, PreprocessingMetadata(
            dropped_features    = self._to_drop,
            imputed_features    = self._imputed_cols,
            target_rows_dropped = target_rows_dropped,
            execution_time_ms   = round(elapsed_ms, 2),
        )

    def transform(
        self, df: pl.DataFrame
    ) -> Tuple[pl.DataFrame, PreprocessingMetadata]:
        if not self._fitted:
            raise RuntimeError("Call fit_transform() before transform().")

        start = time.perf_counter()

        
        target_rows_dropped = 0
        if self.target_column in df.columns:
            before = df.height
            df = self._validate_target(df)
            target_rows_dropped = before - df.height

        df = self._clean_feature_strings(df)

        cols_to_drop = [c for c in self._to_drop if c in df.columns]
        lf = df.lazy().drop(cols_to_drop)

        if self.enable_imputation:
            lf = self._apply_imputer(lf)

        try:
            result_df = lf.collect()
        except Exception as exc:
            logger.error("transform collect() failed: %s", exc)
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000

        return result_df, PreprocessingMetadata(
            dropped_features    = cols_to_drop,
            imputed_features    = self._imputed_cols,
            target_rows_dropped = target_rows_dropped,
            execution_time_ms   = round(elapsed_ms, 2),
        )

    def _validate_target(self, df: pl.DataFrame) -> pl.DataFrame:
        col   = self.target_column
        dtype = df.schema[col]

        if isinstance(dtype, _STRING_DTYPES):
            df = df.with_columns(
                pl.when(
                    pl.col(col).str.strip_chars().str.to_lowercase()
                      .is_in(_SENTINEL_STRINGS)
                )
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            )

        before = df.height
        df     = df.drop_nulls(subset=[col])
        dropped = before - df.height
        if dropped:
            logger.warning("Target '%s': dropped %d/%d rows (null / sentinel values).", col, dropped, before)

        if self.problem_type in (ProblemType.REGRESSION, ProblemType.FORECASTING):
            if not isinstance(df.schema[col], _NUMERIC_DTYPES):
                try:
                    df = df.with_columns(pl.col(col).cast(pl.Float64))
                    logger.info("Target '%s' cast → Float64.", col)
                except Exception as exc:
                    sample = df[col].drop_nulls().head(5).to_list()
                    raise ValueError(
                        f"Target '{col}' cannot be cast to Float64. Sample values: {sample}. Error: {exc}"
                    ) from exc
            elif df.schema[col] != pl.Float64:
                df = df.with_columns(pl.col(col).cast(pl.Float64))

        if df.height == 0:
            raise ValueError(
                f"Target '{col}' cleaning removed ALL rows. "
                "The target column may be entirely null or sentinel values."
            )
        null_remaining = df[col].null_count()
        if null_remaining:
            raise RuntimeError(
                f"Target '{col}' still has {null_remaining} nulls after cleaning."
            )

        return df

    def _clean_feature_strings(self, df: pl.DataFrame) -> pl.DataFrame:
        cols = [
            c for c, t in df.schema.items()
            if isinstance(t, _STRING_DTYPES) and c != self.target_column
        ]
        if not cols:
            return df
        exprs = [
            pl.when(
                pl.col(c).str.strip_chars().str.to_lowercase()
                  .is_in(_SENTINEL_STRINGS)
            )
            .then(None)
            .otherwise(pl.col(c))
            .alias(c)
            for c in cols
        ]
        return df.with_columns(exprs)

    def _classify_columns(self, df: pl.DataFrame) -> None:
        schema = df.schema
        self._numeric_cols  = [
            c for c, t in schema.items()
            if isinstance(t, _NUMERIC_DTYPES) and c != self.target_column
        ]
        self._temporal_cols = [
            c for c, t in schema.items()
            if isinstance(t, _TEMPORAL_DTYPES) and c != self.target_column
        ]
        self._cat_cols      = [
            c for c, t in schema.items()
            if isinstance(t, _STRING_DTYPES) and c != self.target_column
        ]

    def _select_features(
        self, df: pl.DataFrame
    ) -> Tuple[List[str], List[str]]:
        low_var   = self._detect_low_variance(df)
        high_null = self._detect_high_null(df)

        for c in low_var:
            logger.info("Dropping '%s' — low variance (≤ %d unique).", c, self.variance_threshold)
        for c in high_null:
            logger.info("Dropping '%s' — high null ratio (≥ %.0f%%).", c, self.null_threshold * 100)

        return low_var, high_null

    def _detect_low_variance(self, df: pl.DataFrame) -> List[str]:
        all_cols = self._numeric_cols + self._cat_cols + self._temporal_cols
        return [
            c for c in all_cols
            if c not in self.protected_columns and df[c].n_unique() <= self.variance_threshold
        ]

    def _detect_high_null(self, df: pl.DataFrame) -> List[str]:
        if df.height == 0:
            return []
        all_cols = self._numeric_cols + self._cat_cols + self._temporal_cols
        return [
            c for c in all_cols
            if c not in self.protected_columns
            and df[c].null_count() / df.height >= self.null_threshold
        ]

    def _fit_imputer(
        self,
        df:           pl.DataFrame,
        lf:           pl.LazyFrame,
        numeric_cols: List[str],
        cat_cols:     List[str],
    ) -> pl.LazyFrame:
        for col in numeric_cols:
            if col not in df.columns:
                continue
            val = df[col].mean() if self.numeric_strategy == ImputationStrategy.MEAN else df[col].median()
            val_float = float(val) if val is not None else 0.0
            self._numeric_fills[col] = 0.0 if math.isnan(val_float) else val_float

        for col in cat_cols:
            if col not in df.columns:
                continue
            mode = df[col].drop_nulls().mode()
            self._category_fills[col] = mode[0] if mode.len() > 0 else ""

        logger.debug(
            "Imputer fitted: %d numeric fills, %d categorical fills.",
            len(self._numeric_fills), len(self._category_fills),
        )
        return self._apply_imputer(lf)

    def _apply_imputer(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        cols_present = set(lf.columns)
        exprs = []
        for col, val in self._numeric_fills.items():
            if col in cols_present:
                exprs.append(pl.col(col).fill_null(val).fill_nan(val))
        for col, val in self._category_fills.items():
            if col in cols_present:
                exprs.append(pl.col(col).fill_null(val))
        return lf.with_columns(exprs) if exprs else lf