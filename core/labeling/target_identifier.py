from __future__ import annotations
import polars as pl
from typing import Optional, Dict, Any, Set
from dataclasses import dataclass, field
from core.contracts.problem_type import ProblemType


@dataclass(frozen=True)
class TargetDetectionResult:
    target_column: str
    problem_type: ProblemType
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

# --- The Engine ---

class TargetIdentifier:
    """
    Enterprise-grade Target Identification Engine.
    Uses a weighted heuristic matrix to infer the most likely label column.
    """

    # Weighted signals (Enterprise Tuning)
    WEIGHTS = {
        "name_match": 0.50,      # Semantic hint (target, y, etc.)
        "position_bias": 0.15,   # Cultural bias (last column is usually target)
        "cardinality": 0.25,     # Statistical signal
        "type_alignment": 0.10   # Data type suitability
    }

    HINT_KEYWORDS = {"target", "label", "y", "outcome", "class", "price", "churn", "total"}
    FORBIDDEN_KEYWORDS = {"id", "uuid", "index", "timestamp", "created_at"}

    def __init__(self, df: pl.DataFrame,   force_target: Optional[str] = None):
        self.df = df
        self.force_target = force_target
        self.total_rows = df.height

    def identify(self) -> TargetDetectionResult:
        """
        Public API to execute identification logic.
        Formula Name: Multi-Heuristic Aggregation
        """
        if self.total_rows == 0:
            raise ValueError("Target identification requires a non-empty DataFrame.")

        
        if self.force_target:
            if self.force_target not in self.df.columns:
                raise KeyError(f"Forced target '{self.force_target}' not found in DataFrame.")
            return self._build_result(self.force_target, confidence=1.0, source="override")

        # 2. Scoring Phase
        scores = self._calculate_scores()
        if not scores:
            raise ValueError("Zero candidates met the minimum target threshold.")

        # 3. Selection Phase (Sort by highest confidence)
        best_col = max(scores, key=lambda k: scores[k])
        
        return self._build_result(best_col, confidence=scores[best_col], source="inference", all_scores=scores)

    def _calculate_scores(self) -> Dict[str, float]:
        """Iterates through columns and applies the weighting matrix."""
        scores = {}
        cols = self.df.columns

        for i, col in enumerate(cols):
            # Skip forbidden columns (IDs/Timestamps)
            if any(key in col.lower() for key in self.FORBIDDEN_KEYWORDS):
                continue

            score = 0.0
            series = self.df[col]
            unique_count = series.n_unique()
            cardinality_ratio = unique_count / self.total_rows

            # A. Name Signal
            if any(hint in col.lower() for hint in self.HINT_KEYWORDS):
                score += self.WEIGHTS["name_match"]

            # B. Position Signal (Target is usually the last column)
            if i == len(cols) - 1:
                score += self.WEIGHTS["position_bias"]

            # C. Cardinality Signal (Avoid constants and IDs)
            # Ideal targets have variance but aren't unique per row
            if 1 < unique_count < self.total_rows:
                score += self.WEIGHTS["cardinality"]

            # D. Type alignment (Exclude complex types)
            if series.dtype.is_numeric() or series.dtype == pl.Boolean or series.dtype == pl.Utf8:
                score += self.WEIGHTS["type_alignment"]

            if score > 0.1:  # Minimum threshold to be a candidate
                scores[col] = min(score, 1.0)

        return scores

    def _build_result(self, column: str, confidence: float, source: str, **kwargs) -> TargetDetectionResult:
        """Wraps detection with problem type inference."""
        p_type = self._infer_problem_type(column)
        metadata = {"source": source, "total_rows": self.total_rows, **kwargs}
        
        return TargetDetectionResult(
            target_column=column,
            problem_type=p_type,
            confidence=round(confidence, 2),
            metadata=metadata
        )

    def _infer_problem_type(self, column: str) -> ProblemType:
        """
        Refined Problem Type Inference.
        Logic: Prioritizes data types over row count.
        """
        series = self.df[column]
        dtype = series.dtype
        unique = series.n_unique()

        # 1. Forecasting Priority
        has_time_axis = any(t in (pl.Date, pl.Datetime) for t in self.df.dtypes)
        if has_time_axis and dtype.is_numeric():
            return ProblemType.FORECASTING

        # 2. Classification Logic
        if dtype == pl.Boolean or unique == 2:
            return ProblemType.BINARY
        
        if dtype == pl.Utf8 or (dtype.is_integer() and unique <= 20):
            return ProblemType.CLASSIFICATION

        # 3. Regression Logic
        if dtype.is_float() or (dtype.is_integer() and unique > 20):
            return ProblemType.REGRESSION

        return ProblemType.UNKNOWN