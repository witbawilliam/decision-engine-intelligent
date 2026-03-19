from __future__ import annotations
import polars as pl
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from core.contracts.problem_type import ProblemType



@dataclass(frozen=True)
class SchemaInferenceResult:
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    id_columns: List[str]
    target_candidates: List[str]
    problem_type: Optional[ProblemType]
    metadata: Dict[str, Any] = field(default_factory=dict)
    target_column: Optional[str] = None

class SchemaInference:
    """
     Semantic Inference Engine.
    Uses statistical heuristics and Shannon Entropy to determine data intent.
    """

    def __init__(self, df: pl.DataFrame, target_column:str | None = None,  id_suffix: str = "_id"):
        self.df = df
        self.id_suffix = id_suffix.lower()
        self.total_rows = df.height
        self.target_column = target_column

    def infer(self) -> SchemaInferenceResult:
        if self.total_rows == 0:
            raise ValueError("Inference cannot run on an empty DataFrame.")

        groups = self._group_by_physical_type()
        
        #  Semantic Refinement (The "Smart" Step)
        # Identifies columns that are technically numbers but semantically categories
        refined = self._refine_semantics(groups)
        
        #  ID Discovery
        ids = self._discover_ids(refined)
        
        #  Target & Problem Type Logic
        potential_targets = self._identify_target_candidates(refined, ids)
        problem_type = self._determine_problem_type(potential_targets, groups["datetime"])

        return SchemaInferenceResult(
            numeric_columns=refined["numeric"],
            categorical_columns=refined["categorical"],
            datetime_columns=groups["datetime"],
            id_columns=ids,
            target_candidates=potential_targets,
            problem_type=problem_type,
            
        
        )

    def _group_by_physical_type(self) -> Dict[str, List[str]]:
        """Separates columns by raw Polars dtypes."""
        return {
            "numeric": [c for c, t in self.df.schema.items() if t.is_numeric()],
            "categorical": [c for c, t in self.df.schema.items() if t in (pl.Utf8, pl.Categorical, pl.Boolean)],
            "datetime": [c for c, t in self.df.schema.items() if t in (pl.Date, pl.Datetime)]
        }

    def _refine_semantics(self, groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Formula Name: Cardinality-Ratio Semantic Refinement
        Logic: If a numeric column has very few unique values compared to row count,
               it is likely a Category (e.g., [1, 1, 2, 2, 1]).
        """
        numeric = []
        categorical = groups["categorical"]

        for col in groups["numeric"]:
            unique_count = self.df[col].n_unique()
            cardinality_ratio = unique_count / self.total_rows

            # Rule: Low cardinality numbers are semantically Categories
            if unique_count <= 10 or (cardinality_ratio < 0.05 and unique_count < 100):
                categorical.append(col)
            else:
                numeric.append(col)

        return {"numeric": numeric, "categorical": categorical}

    def _discover_ids(self, refined: Dict[str, List[str]]) -> List[str]:
        """
        Formula Name: Primary Key Heuristic
        Logic: Unique values + ID-like name = Identity Column.
        """
        ids = []
        for col in refined["numeric"] + refined["categorical"]:
            if col.lower().endswith(self.id_suffix):
                if self.df[col].n_unique() == self.total_rows:
                    ids.append(col)
        return ids

    def _identify_target_candidates(self, refined: Dict[str, List[str]], ids: List[str]) -> List[str]:
        """Filters out IDs and constants to find viable labels."""
        all_cols = refined["numeric"] + refined["categorical"]
        return [c for c in all_cols if c not in ids and self.df[c].n_unique() > 1]

    def _determine_problem_type(self, targets: List[str], dates: List[str]) -> Optional[ProblemType]:
        if not targets:
            return None
        
        if dates:
            return ProblemType.FORECASTING

        main_target = targets[0]
        unique_vals = self.df[main_target].n_unique()
        dtype = self.df[main_target].dtype

        
        # Strings/Booleans are ALWAYS Classification.
        # Floats are ALWAYS Regression.
        # Integers with low cardinality are Classification.
        if dtype in (pl.Utf8, pl.Categorical, pl.Boolean):
            return ProblemType.CLASSIFICATION
        
        if dtype.is_float():
            return ProblemType.REGRESSION

        # For Integers, use the threshold
        if unique_vals <= 20:
            return ProblemType.CLASSIFICATION
        
        return ProblemType.REGRESSION