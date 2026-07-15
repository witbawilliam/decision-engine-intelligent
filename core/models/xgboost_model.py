from __future__ import annotations
import logging
import time
import joblib
from pathlib import Path
from typing import Optional, Dict, Any, Union
import pandas as pd
import polars as pl
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from dataclasses import dataclass, field
from core.contracts.problem_type import ProblemType


class NotFittedError(RuntimeError):
    """Raised when a model is used for inference before it has been trained."""
    pass


logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ModelResult:
    """The Immutable result of a model inference."""
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    inference_time_ms: float = 0.0

    def __len__(self) -> int:
        """Enables len(model_result)."""
        return len(self.predictions)

    def __getitem__(self, key: int) -> Any:
        """
        Enables result[0] access (Subscripting).
        This redirects bracket access directly to the predictions array.
        """
        return self.predictions[key]

    def to_dict(self) -> Dict[str, Any]:
        """Utility for API serialization."""
        return {
            "predictions": self.predictions.tolist(),
            "probabilities": self.probabilities.tolist() if self.probabilities is not None else None,
            "feature_importance": self.feature_importance,
            "inference_time_ms": self.inference_time_ms
        }
class XGBoostModel:
    """
    Enterprise-grade XGBoost Wrapper.
    Supports native Polars integration, automatic categorical detection, 
    and model persistence.
    """

    def __init__(
        self,
        problem_type: ProblemType,
        params: Optional[Dict[str, Any]] = None,
        model_name: str = "xgboost_v1",
    
        

    ):
        self.problem_type = problem_type
        self.model_name = model_name
        self.params = params or {}
        self.feature_names: Optional[list[str]] = None
        self._is_fitted = False
        self._n_classes: Optional[int] = None
        self.model = self._initialize_model()
        

    def _initialize_model(self) -> Union[XGBClassifier, XGBRegressor]:
        """Initializes the model with production-optimized defaults."""
        # Formulas
        default_params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "enable_categorical": True,  # Critical for Enterprise Data
            "tree_method": "hist",        # Faster for large datasets
        }
        default_params.update(self.params)

        if self.problem_type == ProblemType.CLASSIFICATION:
            return XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                **default_params
            )

        elif self.problem_type == ProblemType.REGRESSION:
            return XGBRegressor(
                objective="reg:squarederror",
                **default_params
            )

        raise ValueError(f"Unsupported problem type: {self.problem_type}")

        

    def fit(self, df: pl.DataFrame, target_column: str) -> XGBoostModel:
        """
        Fits the model using Polars DataFrames directly.
        Includes automatic categorical type conversion for XGBoost.
        """
        logger.info(f"Starting fit for {self.model_name} on target: {target_column}")
        
        if target_column not in df.columns:
            raise KeyError(f"Target '{target_column}' missing from DataFrame.")

        X = df.drop(target_column)
        y = df[target_column]
        self.feature_names = X.columns

        
        X_pd = X.to_pandas()
        for col in X_pd.select_dtypes(["object", "category"]).columns:
            X_pd[col] = X_pd[col].astype("category")

        y_np = y.to_numpy() 

        if self.problem_type == ProblemType.CLASSIFICATION:
            self._n_classes = len(np.unique(y_np))
            if self._n_classes > 2:
                logger.info(
                    f"Multiclass problem detected ({self._n_classes} classes). "
                    "Switching objective to 'multi:softprob'."
                )
                self.model.set_params(
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    num_class=self._n_classes,
                )
            else:
                self._n_classes = 2
                logger.info("Binary classification confirmed.")   

        start_time = time.perf_counter()
        self.model.fit(X_pd, y.to_numpy())
        duration = time.perf_counter() - start_time

        self._is_fitted = True
        logger.info(f"Model fit completed in {duration:.2f} seconds.")
        return self

    def predict(self, df: Any) -> ModelResult:
        if not self._is_fitted:
            raise NotFittedError("Model must be fitted before prediction.")

        if isinstance(df, pd.DataFrame):
            df_pl = pl.from_pandas(df)
        elif isinstance(df, dict):
            df_pl = pl.DataFrame(df)
        elif isinstance(df, pl.DataFrame):
            df_pl = df
        else:
            raise TypeError(f"Inference requires Polars/Pandas DataFrame, got {type(df)}")

    
        X = df_pl.select(self.feature_names)
        X_pd = X.to_pandas()
        
        for col in X_pd.select_dtypes(["object", "category"]).columns:
            X_pd[col] = X_pd[col].astype("category")

        start_time = time.perf_counter()
        predictions = self.model.predict(X_pd)
        
        probabilities = None
        if self.problem_type == ProblemType.CLASSIFICATION:

            probabilities = self.model.predict_proba(X_pd)
            
        inference_time = (time.perf_counter() - start_time) * 1000

        
        return ModelResult(
            predictions=predictions,
            probabilities=probabilities,
            feature_importance=self._get_feature_importance(),
            inference_time_ms=inference_time
        )

    def _get_feature_importance(self) -> Dict[str, float]:
        """Extracts Gain-based importance (Enterprise Standard)."""
        if not self._is_fitted or self.feature_names is None:
            return {}
        

        booster = self.model.get_booster()
        gain_scores = booster.get_score(importance_type="gain")
        
        importance = {}
        for i, name in enumerate(self.feature_names):
            # XGBoost internally uses 'f0', 'f1', ... when feature names are not
            # registered. Fall back gracefully.
            val = gain_scores.get(name) or gain_scores.get(f"f{i}", 0.0)
            importance[name] = float(val)
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    

    
    # PERSISTENCE (The Enterprise Key)
    

    def save(self, path: Union[str, Path]) -> None:
        """Serializes the model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Model saved to {path}")

    @staticmethod
    def load(path: Union[str, Path]) -> XGBoostModel:
        """Loads a model from disk."""
        return joblib.load(path)