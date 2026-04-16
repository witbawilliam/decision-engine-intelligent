
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union

import numpy as np
import polars as pl
import shap
from xgboost import XGBModel

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ExplanationResult:
    
    shap_values: np.ndarray
    shap_importance: Dict[str, float]
    gain_importance: Dict[str, float]
    expected_value: float
    compute_time_ms: float
    feature_names: List[str]

    def to_polars(self) -> pl.DataFrame:
    
        return pl.DataFrame(self.shap_values, schema=self.feature_names)

class XGBExplainer:
    """
    High-Performance Model Explainability Engine.
    Features: Optimized TreeShap, Feature Name Mapping, and Performance Profiling.
    """

    def __init__(
        self, 
        model: XGBModel, 
        feature_names: List[str],
        background_data: Optional[pl.DataFrame] = None
    ):
        self.model = model
        self.feature_names = feature_names
        
        bg_np = background_data.to_numpy() if background_data is not None else None
        
        try:
            self.explainer = shap.TreeExplainer(
                self.model, 
                data=bg_np,
                feature_names=self.feature_names
            )
        except Exception as e:
            logger.error(f"Failed to initialize TreeExplainer: {e}")
            raise

    def explain(self, df: pl.DataFrame, check_additivity: bool = True) -> ExplanationResult:
        """
        Generates global and local explanations with additivity verification.
        """
        start_time = time.perf_counter()
        
        X = df.to_numpy()

        
        try:
            
            shap_output = self.explainer(X, check_additivity=check_additivity)
            
            shap_values = shap_output.values
            expected_value = shap_output.base_values[0] if hasattr(shap_output.base_values, "__len__") else shap_output.base_values

        except Exception as e:
            logger.error(f"Inference explanation failed: {e}")
            raise

        if len(shap_values.shape) == 3:  
            logger.info("Multi-class detected. Reducing to target class index 0.")
            shap_values = shap_values[:, :, 0]

        shap_importance = self._compute_shap_importance(shap_values)
        gain_importance = self._compute_gain_importance()

        duration_ms = (time.perf_counter() - start_time) * 1000
        
        return ExplanationResult(
            shap_values=shap_values,
            shap_importance=shap_importance,
            gain_importance=gain_importance,
            expected_value=float(expected_value),
            compute_time_ms=duration_ms,
            feature_names=self.feature_names
        )

    def _compute_shap_importance(self, shap_values: np.ndarray) -> Dict[str, float]:

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        importance = {
            name: float(val) for name, val in zip(self.feature_names, mean_abs_shap)
        }

        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def _compute_gain_importance(self) -> Dict[str, float]:
        """
        Extracts native XGBoost Gain importance.
        Robustly handles the internal 'f0', 'f1' naming convention.
        """
        booster = self.model.get_booster()
        gain_scores = booster.get_score(importance_type="gain")
        importance = {}
        for i, name in enumerate(self.feature_names):
            val = gain_scores.get(name) or gain_scores.get(f"f{i}", 0.0)
            importance[name] = float(val)
            
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))