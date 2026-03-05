from __future__ import annotations

import logging
import json
from typing import Dict, List, Optional

import numpy as np
import polars as pl

# register types

NUMERIC_DTYPES = {
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                pl.Float32, pl.Float64,
        }



class ManifoldGuard:
    """
    SaaS-grade statistical manifold guard.

    - Serializable
    - Versioned
    - Drift-aware
    - Fast inference
    """

    VERSION = "1.0.0"


   

    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

        self._fitted: bool = False
        self._numeric_columns: List[str] = []
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._feature_count: int = 0

    
    # Training Phase
    

    def fit(self, df: pl.DataFrame):

        if df.is_empty():
            raise ValueError("Cannot fit guard on empty dataset.")

        numeric_cols = [
                    col for col, dtype in df.schema.items()
                    if dtype in NUMERIC_DTYPES
        ]


        if not numeric_cols:
            raise ValueError("No numeric columns found.")

        numeric_df = df.select(numeric_cols)
        matrix = numeric_df.to_numpy()

        self._mean = matrix.mean(axis=0)
        self._std = matrix.std(axis=0)

        # Avoid zero variance explosion
        self._std[self._std < 1e-9] = 1e-9

        self._numeric_columns = numeric_cols
        self._feature_count = len(numeric_cols)
        self._fitted = True

        self._logger.info(
            f"ManifoldGuard v{self.VERSION} fitted on "
            f"{self._feature_count} numeric features."
        )

    
    # Inference Phase
    
    def get_risk_score(self, row: pl.DataFrame) -> float:

        if not self._fitted:
            raise RuntimeError("Guard not fitted or loaded.")

        if row.height != 1:
            raise ValueError("Risk scoring requires a single row.")

        # Align features safely
        try:
            numeric_row = row.select(self._numeric_columns).to_numpy()
        except Exception:
            raise KeyError("Feature mismatch between training and inference schema.")

        standardized = (numeric_row - self._mean) / self._std

        # L2 norm
        distance = np.linalg.norm(standardized)

        # Logistic-style scaling (SaaS-friendly smooth curve)
        risk = 1 - np.exp(-distance / np.sqrt(self._feature_count))

        return float(np.clip(risk, 0.0, 1.0))

    
    # Drift Detection (Batch Mode)
    

    def compute_population_drift(self, df: pl.DataFrame) -> float:
        """
        Returns average drift score across dataset.
        Used for monitoring.
        """

        if not self._fitted:
            raise RuntimeError("Guard not fitted.")

        numeric_df = df.select(self._numeric_columns)
        matrix = numeric_df.to_numpy()

        standardized = (matrix - self._mean) / self._std
        distances = np.linalg.norm(standardized, axis=1)

        avg_distance = distances.mean()

        drift_score = 1 - np.exp(-avg_distance / np.sqrt(self._feature_count))

        return float(np.clip(drift_score, 0.0, 1.0))

    
    # Serialization
    

    def save(self, path: str):
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted guard.")

        payload = {
            "version": self.VERSION,
            "numeric_columns": self._numeric_columns,
            "mean": self._mean.tolist(),
            "std": self._std.tolist(),
            "feature_count": self._feature_count,
        }

        with open(path, "w") as f:
            json.dump(payload, f)

    def load(self, path: str):
        with open(path, "r") as f:
            payload = json.load(f)

        self._numeric_columns = payload["numeric_columns"]
        self._mean = np.array(payload["mean"])
        self._std = np.array(payload["std"])
        self._feature_count = payload["feature_count"]
        self._fitted = True

        self._logger.info(
            f"ManifoldGuard v{payload['version']} loaded "
            f"with {self._feature_count} features."
        )

