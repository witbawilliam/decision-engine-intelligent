from __future__ import annotations

import logging
from typing import Dict, Any
import numpy as np

import polars as pl

from core.pipelines.base_pipeline import BasePipeline
from core.contracts.schema_inference import SchemaInference
from core.contracts.schema_validator import SchemaValidator
from core.feature_engineering.data_quality import DataQualityAnalyzer
from core.feature_engineering.preprocessing_utils import FeatureProcessor, ScalingStrategy,  ImputationStrategy
from core.models.xgboost_model import XGBoostModel
from core.evaluation.regression_metrics import RegressionMetrics
from core.evaluation.feature_importance import XGBExplainer
from core.drift.drift_detector import DriftDetector
from core.contracts.problem_type import ProblemType
from sklearn.preprocessing import LabelEncoder
from core.models.model_registry import ModelRegistry
from core.feature_engineering.tabular_features import TabularIntelligenceEngine
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from storage.s3_client import S3Client

logger = logging.getLogger(__name__)

@dataclass
class PipelineResult:
    status: str
    model_name: str
    model_version: str
    metrics: Dict[str, float]
    artifacts_path: str
    feature_columns: List[str]
    metadata: Dict[str, Any]
    predictions: Optional[List[float]] = None
    actual_values: Optional[List[float]] = None


class TabularPipeline(BasePipeline):

    def __init__(self, df: pl.DataFrame, target_column: str, s3_client: S3Client, problem_type: Optional[str] = None, **kwargs):

        resolved = None
        if problem_type is not None:
            try:
                resolved = ProblemType[problem_type.upper()]
            except KeyError:
                raise ValueError(
                    f"Invalid problem_type '{problem_type}'. "
                    "Must be: regression | classification | forecasting"
                )    
            
        super().__init__(df, target_column, problem_type=resolved, **kwargs)
        self.s3_client = s3_client 
    

    def execute_pipeline(self, model_name: str) -> PipelineResult:
        """
        The entry point for the pipeline execution.
        This orchestrates the private methods in the correct order.
        """
        logger.info(f"Starting pipeline execution for model: {model_name}")

        try:
            # Execution Flow
            self._detect_problem_type()
            self._validate()
            self._feature_engineering()
            self._split()
            self._train()


            self._cached_metrics = self._evaluate_final()
            self._cached_drift   = self._check_drift()


            registration = self.save_to_registry(model_name=model_name)
        
            

            # Construct the Contract
            result = PipelineResult(
                status="SUCCESS",
                model_name=model_name,
                model_version=str(registration.get("version", "unknown")),
                metrics=self._cached_metrics,
                artifacts_path=registration.get("path", ""),
                feature_columns=self.feature_columns,
                metadata={
                    "problem_type": self.problem_type.name,
                    "is_drifted": self._cached_drift["is_drifted"]
                },
                predictions=self.predictions[:20],  
                actual_values=self.actual_values[:20]
            )
            
            logger.info(f"Pipeline finished successfully for {model_name}")
            return result

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise e

    
    
    def _detect_problem_type(self) -> None:
        engine = TabularIntelligenceEngine(df=self.df, target_column=self.target_column)
        self.target_column = engine.target_column

        
        existing = getattr(self, "problem_type", None)
        if existing is not None and isinstance(existing, ProblemType):
            logger.info(f"Problem type provided by caller: {existing.name} — skipping auto-detection.")
            return

        detected_str = engine._detect_problem_type()
        logger.info(f"Auto-detected problem type: {detected_str}")

        mapping = {
            "regression":     ProblemType.REGRESSION,
            "classification": ProblemType.CLASSIFICATION,
        }

        if detected_str in mapping:
            self.problem_type = mapping[detected_str]
        else:
           
            raise ValueError(
                f"Could not determine problem type for target '{self.target_column}'. "
                "Please specify problem_type explicitly in your request "
                "(regression | classification | forecasting)."
            )

        logger.info(f"Target identified: {self.target_column}")
        logger.info(f"Problem type locked: {self.problem_type.name}")


    def _validate(self) -> None:
        """
        Validation flow:
          Detect problem type
          Infer schema
          Validate schema contract
          Data quality audit
        """

        

        # Infer schema
        schema_engine = SchemaInference(
            df=self.df,
            target_column=self.target_column,
        )
        inferred_schema = schema_engine.infer()

        # Validate schema
        validator = SchemaValidator(
            df=self.df,
            schema=inferred_schema,
            problem_type=self.problem_type,
        )
        validator.validate()

        # Data quality
        quality_report = DataQualityAnalyzer(df=self.df).analyze()

        if not quality_report.status:
            raise ValueError("Data quality checks failed.")

        logger.info("Validation completed successfully.")

    
    
    
    def _feature_engineering(self) -> None:
        """
        Feature preprocessing using the production FeatureProcessor.

        Execution order (all inside FeatureProcessor.fit_transform):
          1. Target validation   — sentinel cleanup, null drop, dtype cast
          2. Duplicate removal
          3. Feature string cleaning — sentinels → null in feature cols
          4. Column classification  — numeric / temporal / categorical
          5. Feature selection   — leakage, low-variance, high-null pruning
          6. Imputation          — fit medians/modes on THIS training data
          7. Encoding            — fit ordinal maps on THIS training data
          8. Scaling             — fit stats on THIS training data (NONE for XGBoost)

        The fitted processor is stored on self.processor so predict()
        can call self.processor.transform(inference_df) and reuse the
        exact same statistics — no training-serving skew.
        """
        self.processor = FeatureProcessor(
            target_column    = self.target_column,
            problem_type     = self.problem_type,       # drives target casting + scaling advice
            scaling_strategy = ScalingStrategy.NONE,    # XGBoost is scale-invariant
            leakage_threshold  = 0.995,
            variance_threshold = 1,
            null_threshold     = 0.60,
        )

        processed_df, metadata = self.processor.fit_transform(self.df)

        self.df                   = processed_df
        self.preprocessing_metadata = metadata

        # Derive feature lists from the cleaned schema
        # (columns may have been dropped by the selector)
        self.numeric_features = [
            c for c, t in self.df.schema.items()
            if t.is_numeric() and c != self.target_column
        ]
        self.categorical_features = [
            c for c, t in self.df.schema.items()
            if not t.is_numeric() and c != self.target_column
        ]

        logger.info(
            "Feature engineering complete: %d numeric, %d categorical | "
            "dropped=%d, leaked=%d, imputed=%d, encoded=%d | %.1f ms",
            len(self.numeric_features),
            len(self.categorical_features),
            len(metadata.dropped_features),
            len(metadata.leaked_features),
            len(metadata.imputed_features),
            len(metadata.encoded_features),
            metadata.execution_time_ms,
        )

        if metadata.is_imbalanced:
            logger.warning(
                "Class imbalance detected in target '%s'. "
                "Consider class_weight or oversampling before training.",
                self.target_column,
            )

        if metadata.target_rows_dropped > 0:
            logger.warning(
                "%d rows dropped during target validation "
                "(null or sentinel values in '%s').",
                metadata.target_rows_dropped,
                self.target_column,
            )

    
    
    

    def _split(self) -> None:

        X = self.df.drop(self.target_column).to_pandas()
        y = self.df[self.target_column].to_pandas()

        use_stratify = (
            self.problem_type == ProblemType.CLASSIFICATION
            and y.nunique() <= 50
        )

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y if use_stratify else None
        )

        

        logger.info("Train-test split completed.")

    
    
    

    def _train(self) -> None:
        self.model = XGBoostModel(problem_type=self.problem_type)

        y_train = self.y_train

        if self.problem_type == ProblemType.CLASSIFICATION:
            self.label_encoder = LabelEncoder()
            y_train = self.label_encoder.fit_transform(self.y_train)
        else: 
            self.label_encoder = None

        train_df = pl.from_pandas(self.X_train)
        train_df = train_df.with_columns(
            pl.Series(self.target_column, y_train)
        )

        self.model.fit(
            df=train_df,
            target_column=self.target_column
        )

        self.feature_columns = self.model.feature_names
        logger.info("Model training completed.")

    

    def _evaluate_final(self) -> Dict[str, float]:
        assert hasattr(self, "model"), "_train() must be called before _evaluate_final()"
        test_df = pl.from_pandas(self.X_test)
        test_df = test_df.with_columns(
            pl.Series(self.target_column, self.y_test)
        )

        result = self.model.predict(test_df.drop(self.target_column))
        predictions = result.predictions

        self.predictions = predictions
        self.actual_values = self.y_test.tolist()

        if self.problem_type == ProblemType.REGRESSION:
                evaluator = RegressionMetrics()
                metric_result = evaluator.evaluate(
                y_true=self.y_test,
                y_pred=predictions,
                n_features=len(self.feature_columns)
            )

                return metric_result.to_dict()

        
     
        
        if self.problem_type == ProblemType.CLASSIFICATION:

            
            y_true = (

                self.label_encoder.transform(self.y_test)
                if self.label_encoder is not None
                else self.y_test
            )

            metrics = {
                "accuracy": float(accuracy_score(y_true, predictions)),
                "f1_weighted": float(f1_score(y_true, predictions, average="weighted")),
                "precision": float(precision_score(y_true, predictions, average="weighted"))
            }
            logger.info(f"Classification Metrics: {metrics}")
            return metrics
    
        raise NotImplementedError(
            f"Metric evaluation not implemented for problem type: {self.problem_type}"
        )
    

    def predict(self, data: Any) -> Any:
        """
        High-level inference entry point.
 
        Uses the FITTED FeatureProcessor from training (self.feature_processor)
        to guarantee that all transformations are applied with the exact same
        statistics computed during training — eliminating training-serving skew.
        """
        assert hasattr(self, "processor"), (
            "Pipeline has not been trained. Call execute_pipeline() first."
        )
        assert hasattr(self, "model"), (
            "Pipeline has not been trained. Call execute_pipeline() first."
        )
 
        # Normalise input to Polars
        if isinstance(data, pl.DataFrame):
            inference_df = data
        elif isinstance(data, pd.DataFrame):
            inference_df = pl.from_pandas(data)
        elif isinstance(data, dict):
            inference_df = pl.DataFrame(data)
        else:
            raise TypeError(f"predict() expects a DataFrame or dict, got {type(data)}")
 
        
        processed_df, _ = self.processor.process(inference_df)
 
        return self.model.predict(processed_df.drop(self.target_column, strict=False))
    
    
    def _post_training_analysis(self) -> Dict[str, Any]:
        analyzer = XGBExplainer(
            self.model.model,   
            feature_names=self.feature_columns
        )

        importance_report = analyzer.explain(pl.from_pandas(self.X_train))

        return importance_report
        

    def _check_drift(self) -> Dict[str, Any]:
        detector = DriftDetector()

        drift_report = detector.check_drift(
            reference_df=pl.from_pandas(self.X_train),
            current_df=pl.from_pandas(self.X_test),
        )

        return {
            "is_drifted": drift_report.is_drifted,
            "flagged_features": drift_report.flagged_features,
        }


    


    def _collect_artifacts(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "label_encoder": self.label_encoder,          
            "feature_processor": self.processor,
            "problem_type": self.problem_type.name,
            "feature_importance": self._post_training_analysis(),
            "drift_report": self._check_drift(),
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "preprocessing_metadata": self.preprocessing_metadata,
        }
    

    def save_to_registry(self, model_name: str, registry_path: str = "ml_registry"):
        """
        Connects the Pipeline results to the Model Registry.
        """

        if not self.s3_client:
            raise ValueError("Pipeline initialized without an S3Client. Cannot register model.")
        
        registry = ModelRegistry(base_path=registry_path, s3_client=self.s3_client)
        
        # Collect artifacts and metrics
        artifacts = self._collect_artifacts()
    
        raw_params = self.model.model.get_params() if hasattr(self.model.model, 'get_params') else {}

    
        sanitized_params = {
            k: (None if isinstance(v, float) and np.isnan(v) else v) 
            for k, v in raw_params.items()
        }
        
        
        registration_result = registry.register(
            model=self.model,
            model_name=model_name,
            metrics=self._cached_metrics,
            parameters=self.model.model.get_params() if hasattr(self.model.model, 'get_params') else {},
            problem_type=self.problem_type.name,
            stage="staging"
        )

        logger.info(f"Model successfully registered: {model_name} v{registration_result['version']}")
        return registration_result