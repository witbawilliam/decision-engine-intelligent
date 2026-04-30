from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import polars as pl
import shap
from xgboost import XGBModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExplanationResult:
    """
    Immutable result of a SHAP explanation run.

    shap_values shape:
      - Binary classification / regression: (n_samples, n_features)
      - Multiclass: (n_samples, n_features, n_classes)  ← full tensor preserved
    """
    shap_values: np.ndarray
    shap_importance: Dict[str, float]
    gain_importance: Dict[str, float]
    expected_value: float
    compute_time_ms: float
    feature_names: List[str]
    n_classes: int = 1  # 1 = regression/binary; >1 = multiclass

    def to_polars(self) -> pl.DataFrame:
        """
        Returns a Polars DataFrame of SHAP values.
        For multiclass, returns values for class index 0 only.
        Call to_polars_multiclass() for the full tensor.
        """
        values = (
            self.shap_values[:, :, 0]
            if len(self.shap_values.shape) == 3
            else self.shap_values
        )
        return pl.DataFrame(values, schema=self.feature_names)

    def to_polars_multiclass(self, class_index: int) -> pl.DataFrame:
        """Returns SHAP values for a specific class (multiclass only)."""
        if len(self.shap_values.shape) != 3:
            raise ValueError("to_polars_multiclass() is only valid for multiclass problems.")
        return pl.DataFrame(self.shap_values[:, :, class_index], schema=self.feature_names)


class XGBExplainer:
    """
    High-Performance Model Explainability Engine.
    Features: Optimised TreeSHAP, correct multiclass handling,
    gain importance, and performance profiling.
    """

    def __init__(
        self,
        model: XGBModel,
        feature_names: List[str],
        background_data: Optional[pl.DataFrame] = None,
    ):
        self.model = model
        self.feature_names = feature_names

        # FIX #10: Convert background data through pandas to preserve dtypes
        # (categorical columns would be lost with .to_numpy() directly on Polars).
        bg_pd = None
        if background_data is not None:
            bg_pd = background_data.to_pandas()
            for col in bg_pd.select_dtypes(["object", "category"]).columns:
                bg_pd[col] = bg_pd[col].astype("category")

        try:
            self.explainer = shap.TreeExplainer(
                self.model,
                data=bg_pd,          # pandas keeps dtype semantics intact
                feature_names=self.feature_names,
            )
        except Exception as e:
            logger.error(f"Failed to initialise TreeExplainer: {e}")
            raise

    def explain(
        self,
        df: pl.DataFrame,
        check_additivity: bool = True,
    ) -> ExplanationResult:
        """
        Generates global and local SHAP explanations.

        Multiclass behaviour (FIX #11):
        - The full (n_samples, n_features, n_classes) SHAP tensor is preserved
          in ExplanationResult.shap_values instead of silently being sliced to
          class 0 only.
        - shap_importance is computed as the mean across ALL classes so it
          represents global feature impact for the whole model.
        - A WARNING is logged so engineers know multiclass is active.
        """
        start_time = time.perf_counter()

        # FIX #10: Convert to pandas (not numpy) to preserve categorical dtypes.
        X_pd = df.to_pandas()
        for col in X_pd.select_dtypes(["object", "category"]).columns:
            X_pd[col] = X_pd[col].astype("category")

        try:
            shap_output = self.explainer(X_pd, check_additivity=check_additivity)
            shap_values = shap_output.values
            expected_value = (
                float(shap_output.base_values[0])
                if hasattr(shap_output.base_values, "__len__")
                else float(shap_output.base_values)
            )
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            raise

        # Determine multiclass status
        is_multiclass = len(shap_values.shape) == 3
        n_classes = shap_values.shape[2] if is_multiclass else 1

        if is_multiclass:
            # FIX #11: Do NOT silently drop all classes except 0.
            # Log a clear warning and preserve the full tensor.
            logger.warning(
                f"Multiclass problem detected ({n_classes} classes). "
                "ExplanationResult.shap_values contains the full "
                "(n_samples, n_features, n_classes) tensor. "
                "shap_importance is averaged across all classes. "
                "Use ExplanationResult.to_polars_multiclass(class_index) "
                "to inspect per-class values."
            )

        shap_importance = self._compute_shap_importance(shap_values, is_multiclass)
        gain_importance = self._compute_gain_importance()
        duration_ms = (time.perf_counter() - start_time) * 1000

        return ExplanationResult(
            shap_values=shap_values,          # full tensor — no data lost
            shap_importance=shap_importance,
            gain_importance=gain_importance,
            expected_value=expected_value,
            compute_time_ms=duration_ms,
            feature_names=self.feature_names,
            n_classes=n_classes,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_shap_importance(
        self,
        shap_values: np.ndarray,
        is_multiclass: bool,
    ) -> Dict[str, float]:
        """
        Computes mean |SHAP| importance across all samples (and all classes
        for multiclass) so the result represents total model-level impact.
        """
        if is_multiclass:
            # Shape: (n_samples, n_features, n_classes)
            # Mean over samples (axis=0) then mean over classes (axis=-1)
            mean_abs_shap = np.abs(shap_values).mean(axis=0).mean(axis=-1)
        else:
            # Shape: (n_samples, n_features)
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

        importance = {
            name: float(val)
            for name, val in zip(self.feature_names, mean_abs_shap)
        }
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def _compute_gain_importance(self) -> Dict[str, float]:
        """
        Extracts native XGBoost Gain importance via the booster API.
        Handles the internal 'f0', 'f1', ... naming convention robustly.
        """
        booster = self.model.get_booster()
        gain_scores = booster.get_score(importance_type="gain")
        importance = {}
        for i, name in enumerate(self.feature_names):
            val = gain_scores.get(name) or gain_scores.get(f"f{i}", 0.0)
            importance[name] = float(val)
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))