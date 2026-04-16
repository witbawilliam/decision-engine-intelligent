from __future__ import annotations
import polars as pl
from enum import Enum
from typing import List, Dict, Optional, Any, Protocol
from pydantic import BaseModel, Field, ConfigDict, model_validator
from dataclasses import dataclass, field
from core.contracts.problem_type import ProblemType




#  Structured Results & Exceptions 

class SchemaViolationError(Exception):
    """Exception raised when one or more contract violations occur."""
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Schema Validation Failed: {errors}")

@dataclass(frozen=True)
class SchemaValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)
    
#  Data Contracts (The "Expectation") 

class ColumnContract(BaseModel):
    name: str
    dtype: Any  
    nullable: bool = False
    unique: bool = False

class DatasetSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    columns: Dict[str, ColumnContract]
    target_column: str
    dataset_name: Optional[str] = "Default_Dataset"

    @model_validator(mode='before')
    @classmethod
    def canonical_normalization(cls, data: Any) -> Dict[str, Any]:
        """Standardizes list or dict input into a unified map."""
        if isinstance(data, dict) and isinstance(data.get("columns"), list):
            data["columns"] = {col['name']: col for col in data["columns"]}
        return data

#  Validation Rules (The "Strategy") 

class ValidationRule(Protocol):
    """Interface for a single validation check."""
    def verify(self, df: pl.DataFrame, schema: DatasetSchema, p_type: ProblemType) -> Optional[str]:
        ...

class EmptyDatasetRule:
    def verify(self, df: pl.DataFrame, *_) -> Optional[str]:
        if df.height == 0:
          return "Critical: Input DataFrame is empty." 
        else:
            return None

class TargetIntegrityRule:
    def verify(self, df: pl.DataFrame, schema: DatasetSchema, *_) -> Optional[str]:
        if schema.target_column not in df.columns:
            return f"Missing Column: Target '{schema.target_column}' not found in data."
        return None

class VarianceRule:
    def verify(self, df: pl.DataFrame, schema: DatasetSchema, p_type: ProblemType) -> Optional[str]:
        """Formula: |U|_y > 1. Checks for constant targets."""
        if p_type in (ProblemType.REGRESSION, ProblemType.CLASSIFICATION):
            if schema.target_column in df.columns:
                if df.select(pl.col(schema.target_column).n_unique()).item() <= 1:
                    return f"Zero Variance: Target '{schema.target_column}' has only one unique value."
        return None

class TemporalRule:
    def verify(self, df: pl.DataFrame, _, p_type: ProblemType) -> Optional[str]:
        if p_type == ProblemType.FORECASTING:
            has_date = any(t in (pl.Date, pl.Datetime) for t in df.dtypes)
            if not has_date:
                return "Type Error: Forecasting requires at least one Date/Datetime column."
        return None

# Orchestrator (The "Validator") 

class SchemaValidator:
    """
     validator that evaluates a DataFrame against a DatasetSchema.
    """

    def __init__(self, df: pl.DataFrame, schema: DatasetSchema, problem_type: ProblemType):
        self._df = df
        self._schema = schema
        self._type = problem_type
        # Registry of rules to execute
        self._rules: List[ValidationRule] = [
            EmptyDatasetRule(),
            TargetIntegrityRule(),
            VarianceRule(),
            TemporalRule()
        ]

    def validate(self, raise_on_failure: bool = False) -> SchemaValidationResult:
        """
        Executes all rules and aggregates failures into a single report.
        """
        errors = []
        for rule in self._rules:
            error = rule.verify(self._df, self._schema, self._type)
            if error:
                errors.append(error)

        is_valid = len(errors) == 0

        if not is_valid and raise_on_failure:
            raise SchemaViolationError(errors)

        return SchemaValidationResult(valid=is_valid, errors=errors)