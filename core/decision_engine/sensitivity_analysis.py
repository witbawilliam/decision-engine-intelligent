from __future__ import annotations

import logging
from typing import Dict, Any, List, Protocol

import polars as pl
import numpy as np
from pydantic import BaseModel, Field, ConfigDict



# Model Interface


class PredictModel(Protocol):
    def predict(self, data: pl.DataFrame) -> Any:
        ...



# Response Schema


class FeatureSensitivity(BaseModel):
    model_config = ConfigDict(frozen=True)

    feature: str
    baseline_value: float
    sensitivity_score: float  # Normalized importance
    max_prediction_shift: float


class SensitivityResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    baseline_prediction: float
    feature_rankings: List[FeatureSensitivity]



# Sensitivity Engine


class SensitivityAnalyzer:
    """
    Production-grade local sensitivity analysis engine.

    Uses finite difference perturbation.
    Model-agnostic.
    """

    def __init__(
        self,
        model: PredictModel,
        perturbation_ratio: float = 0.05,  # 5% default
    ):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._model = model
        self._ratio = float(perturbation_ratio)

        if not hasattr(model, "predict"):
            raise TypeError("Model must implement predict(pl.DataFrame)")

    
    # Public API

    def analyze(self, row: pl.DataFrame) -> SensitivityResult:

        if row.height != 1:
            raise ValueError("Sensitivity analysis requires a single row.")

        baseline_prediction = self._predict(row)

        feature_scores = []

        for column in row.columns:
            if not pl.datatypes.is_numeric(row.schema[column]):
                continue  # Skip non-numeric safely

            base_val = float(row.get_column(column)[0])

            if base_val == 0:
                delta = self._ratio
            else:
                delta = abs(base_val) * self._ratio

            perturbed_up = row.with_columns(
                pl.lit(base_val + delta).alias(column)
            )

            perturbed_down = row.with_columns(
                pl.lit(base_val - delta).alias(column)
            )

            pred_up = self._predict(perturbed_up)
            pred_down = self._predict(perturbed_down)

            max_shift = max(
                abs(pred_up - baseline_prediction),
                abs(pred_down - baseline_prediction),
            )

            feature_scores.append(
                {
                    "feature": column,
                    "baseline_value": base_val,
                    "max_shift": max_shift,
                }
            )

        if not feature_scores:
            raise ValueError("No numeric features available for sensitivity analysis.")

        # Normalize scores

        max_global_shift = max(f["max_shift"] for f in feature_scores) + 1e-9

        results = []

        for f in feature_scores:
            normalized_score = f["max_shift"] / max_global_shift

            results.append(
                FeatureSensitivity(
                    feature=f["feature"],
                    baseline_value=f["baseline_value"],
                    sensitivity_score=float(normalized_score),
                    max_prediction_shift=float(f["max_shift"]),
                )
            )

        # Sort descending

        results.sort(key=lambda x: x.sensitivity_score, reverse=True)

        return SensitivityResult(
            baseline_prediction=float(baseline_prediction),
            feature_rankings=results,
        )

    # Internal Prediction Wrapper
    
    def _predict(self, row: pl.DataFrame) -> float:
        raw = self._model.predict(row)

        if not hasattr(raw, "__len__") or len(raw) == 0:
            raise RuntimeError("Model returned empty prediction.")

        return float(raw[0])
