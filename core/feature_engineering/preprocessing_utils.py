
from __future__ import annotations

import logging
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


class ScalingStrategy(str, Enum):
    STANDARD = "standard"   # (x − mean) / std          sensitive to outliers
    MINMAX   = "minmax"     # (x − min)  / (max − min)  bounded [0, 1]
    ROBUST   = "robust"     # (x − median) / IQR        outlier-resistant ✓
    NONE     = "none"       # pass-through (trees don't need scaling)



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
    """Immutable audit trail returned alongside every transformed DataFrame."""
    dropped_features:    List[str] = field(default_factory=list)
    leaked_features:     List[str] = field(default_factory=list)
    scaled_features:     List[str] = field(default_factory=list)
    encoded_features:    List[str] = field(default_factory=list)
    imputed_features:    List[str] = field(default_factory=list)
    duplicates_removed:  int       = 0
    target_rows_dropped: int       = 0
    is_imbalanced:       bool      = False
    execution_time_ms:   float     = 0.0



class FeatureProcessor:
   

    def __init__(
        self,
        target_column:        str,
        problem_type:         Optional[ProblemType]  = None,
        numeric_strategy:     ImputationStrategy     = ImputationStrategy.MEDIAN,
        categorical_strategy: ImputationStrategy     = ImputationStrategy.MODE,
        scaling_strategy:     ScalingStrategy        = ScalingStrategy.ROBUST,
        variance_threshold:   int                    = 1,
        leakage_threshold:    float                  = 0.995,
        null_threshold:       float                  = 0.60,
    ) -> None:
        # Config
        self.target_column        = target_column
        self.problem_type         = problem_type
        self.numeric_strategy     = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.scaling_strategy     = scaling_strategy
        self.variance_threshold   = variance_threshold
        self.leakage_threshold    = leakage_threshold
        self.null_threshold       = null_threshold

        # Fitted state — populated by fit_transform(), reused by transform()
        self._fitted         = False
        self._to_drop:       List[str]                    = []
        self._numeric_cols:  List[str]                    = []
        self._cat_cols:      List[str]                    = []
        self._temporal_cols: List[str]                    = []
        self._numeric_fills:  Dict[str, float]            = {}
        self._category_fills: Dict[str, Any]              = {}
        self._encoding_maps: Dict[str, Dict[Any, int]]    = {}
        self._scaling_params: Dict[str, Dict[str, float]] = {}
        # Audit
        self._scaled_cols:   List[str]                    = []
        self._encoded_cols:  List[str]                    = []
        self._imputed_cols:  List[str]                    = []

   

    def fit_transform(
        self, df: pl.DataFrame
    ) -> Tuple[pl.DataFrame, PreprocessingMetadata]:
        """
        Fit on training data, return the cleaned DataFrame + audit metadata.
        Must be called exactly once per pipeline run.
        """
        if self._fitted:
            raise RuntimeError(
                "FeatureProcessor is already fitted. "
                "Create a new instance for each training run."
            )
        start        = time.perf_counter()
        removed_rows = 0

        target_rows_dropped = 0
        if self.target_column in df.columns:
            before = df.height
            df = self._validate_target(df)
            target_rows_dropped = before - df.height

        df, removed_rows = self._remove_duplicates(df)

        df = self._clean_feature_strings(df)

        self._classify_columns(df)

        leaked, low_var, high_null = self._select_features(df)
        self._to_drop = list(set(leaked + low_var + high_null))

        active_numeric  = [c for c in self._numeric_cols  if c not in self._to_drop]
        active_cat      = [c for c in self._cat_cols      if c not in self._to_drop]
        active_temporal = [c for c in self._temporal_cols if c not in self._to_drop]

        lf = df.lazy().drop(self._to_drop)

        lf = self._fit_imputer(df, lf, active_numeric, active_cat + active_temporal)

        lf = self._fit_encoder(df, lf, active_cat, active_temporal)

        lf = self._fit_scaler(df, lf, active_numeric)

        
        try:
            result_df = lf.collect()
        except Exception as exc:
            logger.error("fit_transform collect() failed: %s", exc)
            raise

        # Persist active column lists for transform()
        self._numeric_cols  = active_numeric
        self._cat_cols      = active_cat
        self._temporal_cols = active_temporal
        self._scaled_cols   = list(self._scaling_params.keys())
        self._encoded_cols  = list(self._encoding_maps.keys())
        self._imputed_cols  = (
            list(self._numeric_fills.keys()) +
            list(self._category_fills.keys())
        )
        self._fitted = True

        is_imbalanced = self._check_imbalance(df)
        elapsed_ms    = (time.perf_counter() - start) * 1000

        logger.info(
            "fit_transform complete — %d rows, %d numeric, %d cat, "
            "%d dropped, %.1f ms",
            result_df.height, len(active_numeric), len(active_cat),
            len(self._to_drop), elapsed_ms,
        )

        return result_df, PreprocessingMetadata(
            dropped_features    = self._to_drop,
            leaked_features     = leaked,
            scaled_features     = self._scaled_cols,
            encoded_features    = self._encoded_cols,
            imputed_features    = self._imputed_cols,
            duplicates_removed  = removed_rows,
            target_rows_dropped = target_rows_dropped,
            is_imbalanced       = is_imbalanced,
            execution_time_ms   = round(elapsed_ms, 2),
        )

    def transform(
        self, df: pl.DataFrame
    ) -> Tuple[pl.DataFrame, PreprocessingMetadata]:
        """
        Apply fitted transformations to inference data.
        Uses training statistics — never recomputes from the inference batch.
        """
        if not self._fitted:
            raise RuntimeError("Call fit_transform() before transform().")

        start = time.perf_counter()

        df = self._clean_feature_strings(df)

        cols_to_drop = [c for c in self._to_drop if c in df.columns]
        lf = df.lazy().drop(cols_to_drop)

        lf = self._apply_imputer(lf)
        lf = self._apply_encoder(lf)
        lf = self._apply_scaler(lf)

        try:
            result_df = lf.collect()
        except Exception as exc:
            logger.error("transform collect() failed: %s", exc)
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000

        return result_df, PreprocessingMetadata(
            dropped_features  = cols_to_drop,
            scaled_features   = self._scaled_cols,
            encoded_features  = self._encoded_cols,
            imputed_features  = self._imputed_cols,
            execution_time_ms = round(elapsed_ms, 2),
        )

    @property
    def scaling_params(self) -> Dict[str, Dict[str, float]]:
        """Expose fitted scaler params for model registry serialisation."""
        return dict(self._scaling_params)

    

    def _validate_target(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Clean and cast the target column.
          - Replace sentinel strings with null
          - Drop null-target rows
          - Cast to Float64 for regression/forecasting
          - Validate result is non-empty and null-free
        """
        col   = self.target_column
        dtype = df.schema[col]

        # Replace sentinels → null (string targets only)
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

        # Drop rows where target is null
        before = df.height
        df     = df.drop_nulls(subset=[col])
        dropped = before - df.height
        if dropped:
            logger.warning(
                "Target '%s': dropped %d/%d rows (null / sentinel values).",
                col, dropped, before,
            )

        # Cast to numeric for regression and forecasting
        if self.problem_type in (ProblemType.REGRESSION, ProblemType.FORECASTING):
            if not isinstance(df.schema[col], _NUMERIC_DTYPES):
                try:
                    df = df.with_columns(pl.col(col).cast(pl.Float64))
                    logger.info("Target '%s' cast → Float64.", col)
                except Exception as exc:
                    sample = df[col].drop_nulls().head(5).to_list()
                    raise ValueError(
                        f"Target '{col}' cannot be cast to Float64. "
                        f"Sample values: {sample}. Error: {exc}"
                    ) from exc
            elif df.schema[col] != pl.Float64:
                df = df.with_columns(pl.col(col).cast(pl.Float64))

        # Final guard
        if df.height == 0:
            raise ValueError(
                f"Target '{col}' cleaning removed ALL rows. "
                "The target column may be entirely null or sentinel values."
            )
        null_remaining = df[col].null_count()
        if null_remaining:
            raise RuntimeError(
                f"Target '{col}' still has {null_remaining} nulls after cleaning — "
                "this is a bug, please report it."
            )

        return df

    

    def _remove_duplicates(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, int]:
        clean   = df.unique(maintain_order=True)
        removed = df.height - clean.height
        if removed:
            logger.info("Removed %d duplicate rows.", removed)
        return clean, removed

   

    def _clean_feature_strings(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Replace sentinel strings in all string feature columns with null.
        Imputation will fill them with the training mode/median.
        Target column is excluded — handled by _validate_target().
        """
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
    ) -> Tuple[List[str], List[str], List[str]]:
        leaked      = self._detect_leakage(df)
        low_var     = self._detect_low_variance(df)
        high_null   = self._detect_high_null(df)

        for c in leaked:
            logger.warning("Dropping '%s' — leakage (|corr| ≥ %.3f).", c, self.leakage_threshold)
        for c in low_var:
            logger.info("Dropping '%s' — low variance (≤ %d unique).", c, self.variance_threshold)
        for c in high_null:
            logger.info("Dropping '%s' — high null ratio (≥ %.0f%%).", c, self.null_threshold * 100)

        return leaked, low_var, high_null

    def _detect_leakage(self, df: pl.DataFrame) -> List[str]:
        target = self.target_column
        if target not in df.columns or not isinstance(df.schema[target], _NUMERIC_DTYPES):
            return []
        leaked = []
        for col in self._numeric_cols:
            try:
                corr = df.select(pl.corr(col, target)).item()
                if corr is not None and abs(corr) >= self.leakage_threshold:
                    leaked.append(col)
            except Exception:
                pass
        return leaked

    def _detect_low_variance(self, df: pl.DataFrame) -> List[str]:
        all_cols = self._numeric_cols + self._cat_cols + self._temporal_cols
        return [c for c in all_cols if df[c].n_unique() <= self.variance_threshold]

    def _detect_high_null(self, df: pl.DataFrame) -> List[str]:
        if df.height == 0:
            return []
        all_cols = self._numeric_cols + self._cat_cols + self._temporal_cols
        return [
            c for c in all_cols
            if df[c].null_count() / df.height >= self.null_threshold
        ]

    
    def _fit_imputer(
        self,
        df:           pl.DataFrame,
        lf:           pl.LazyFrame,
        numeric_cols: List[str],
        cat_cols:     List[str],
    ) -> pl.LazyFrame:
        """Compute fill values from training data and store on self."""
        for col in numeric_cols:
            if col not in df.columns:
                continue
            if self.numeric_strategy == ImputationStrategy.MEAN:
                val = df[col].mean()
            else:
                val = df[col].median()      # default: MEDIAN
            self._numeric_fills[col] = float(val) if val is not None else 0.0

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
        """Apply stored fill values — used by both fit_transform and transform."""
        exprs = []
        for col, val in self._numeric_fills.items():
            exprs.append(pl.col(col).fill_null(val).fill_nan(val))
        for col, val in self._category_fills.items():
            exprs.append(pl.col(col).fill_null(val))
        return lf.with_columns(exprs) if exprs else lf

    

    def _fit_encoder(
        self,
        df:           pl.DataFrame,
        lf:           pl.LazyFrame,
        cat_cols:     List[str],
        temporal_cols: List[str],
    ) -> pl.LazyFrame:
        """Build ordinal maps from training data and store on self."""
        # Convert temporals → strings in the reference df for consistent map keys
        df_ref = df
        for col in temporal_cols:
            if col in df_ref.columns:
                df_ref = df_ref.with_columns(
                    pl.col(col).dt.to_string("%Y-%m-%d").alias(col)
                )

        all_cat = cat_cols + temporal_cols
        for col in all_cat:
            if col not in df_ref.columns:
                continue
            uniques = df_ref[col].drop_nulls().unique().to_list()
            self._encoding_maps[col] = {
                v: i for i, v in enumerate(sorted(str(u) for u in uniques))
            }

        logger.debug("Encoder fitted: %d columns.", len(self._encoding_maps))
        return self._apply_encoder(lf)

    def _apply_encoder(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Apply stored ordinal maps — unseen categories → -1 (safe for trees)."""
        # Convert temporal columns to strings first
        if self._temporal_cols:
            active = [c for c in self._temporal_cols if c in lf.columns]
            if active:
                lf = lf.with_columns([
                    pl.col(c).dt.to_string("%Y-%m-%d") for c in active
                ])

        exprs = []
        for col, mapping in self._encoding_maps.items():
            exprs.append(
                pl.col(col)
                .cast(pl.Utf8)
                .replace(mapping, default=-1)
                .cast(pl.Int32)
                .alias(col)
            )
        return lf.with_columns(exprs) if exprs else lf

    

    def _fit_scaler(
        self,
        df:           pl.DataFrame,
        lf:           pl.LazyFrame,
        numeric_cols: List[str],
    ) -> pl.LazyFrame:
    
  
        
        if self.scaling_strategy == ScalingStrategy.NONE:
            return lf

        for col in numeric_cols:
            if col not in df.columns:
                continue
            s = df[col].drop_nulls()

            if self.scaling_strategy == ScalingStrategy.STANDARD:
                loc   = float(s.mean()   or 0.0)
                scale = float(s.std()    or 1.0) or 1.0

            elif self.scaling_strategy == ScalingStrategy.MINMAX:
                loc   = float(s.min()    or 0.0)
                scale = float((s.max() or 1.0) - loc) or 1.0

            elif self.scaling_strategy == ScalingStrategy.ROBUST:
                loc   = float(s.median() or 0.0)
                q75   = float(s.quantile(0.75) or 1.0)
                q25   = float(s.quantile(0.25) or 0.0)
                scale = (q75 - q25) or 1.0

            else:
                raise ValueError(f"Unknown scaling strategy: {self.scaling_strategy}")

            self._scaling_params[col] = {"loc": loc, "scale": scale}

        logger.info(
            "Scaler fitted [%s]: %d numeric columns.",
            self.scaling_strategy.value, len(self._scaling_params),
        )
        return self._apply_scaler(lf)

    def _apply_scaler(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Apply stored scaling params — used by both fit_transform and transform."""
        if self.scaling_strategy == ScalingStrategy.NONE or not self._scaling_params:
            return lf
        exprs = [
            ((pl.col(col) - p["loc"]) / p["scale"]).alias(col)
            for col, p in self._scaling_params.items()
        ]
        return lf.with_columns(exprs)

    

    def _check_imbalance(self, df: pl.DataFrame) -> bool:
        """Flag if the dominant class exceeds 80% of the target distribution."""
        if self.target_column not in df.columns:
            return False
        counts_df = df[self.target_column].value_counts()
        if counts_df.height < 2:
            return False
        count_col = "count" if "count" in counts_df.columns else counts_df.columns[1]
        return (counts_df[count_col].max() / df.height) >= 0.80