from __future__ import annotations
import io
import logging
from typing import Dict, Any
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
import os

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
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from storage.s3_client import S3Client
from storage.postgres_client import PostgresClient

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
    artifacts: Dict[str, Any] = field(default_factory=dict)


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

        self._cached_metrics: Dict[str, float] = {}
        self._cached_drift: Dict[str, Any] = {}
        self._cached_importance: Dict[str, Any] = {}
    

    def execute_pipeline(self, model_name: str) -> PipelineResult:
            """
            The entry point for the pipeline execution.
            This orchestrates the private methods in the correct order.
            """
            logger.info(f"Starting pipeline execution for model: {model_name}")

            try:
                
                self._detect_problem_type()
                self._validate()
                self._feature_engineering()
                self._split()
                self._train()

                self._cached_metrics = self._evaluate_final()
                self._cached_drift   = self._check_drift()
                self._cached_importance = self._post_training_analysis()

                registration = self.save_to_registry(model_name=model_name)

                self._save_predictions_to_db(                   
                    model_name=registration.get("model_name", model_name),
                    model_version=str(registration.get("version", "unknown"))
                )
                
                
                safe_predictions = []
                if self.predictions is not None:
                    safe_predictions = getattr(self.predictions[:20], "tolist", lambda: list(self.predictions[:20]))()

                safe_actual_values = []
                if self.actual_values is not None:
                    safe_actual_values = getattr(self.actual_values[:20], "tolist", lambda: list(self.actual_values[:20]))()
                

                
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
                    predictions=safe_predictions,    
                    actual_values=safe_actual_values    
                )

                
                result.artifacts = {
                    "predictions": safe_predictions,
                    "actual_values": safe_actual_values
                }
                
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

        
        schema_engine = SchemaInference(
            df=self.df,
            target_column=self.target_column,
        )
        inferred_schema = schema_engine.infer()

        validator = SchemaValidator(
            df=self.df,
            schema=inferred_schema,
            problem_type=self.problem_type,
        )
        validator.validate()

        quality_report = DataQualityAnalyzer(df=self.df).analyze()

        if not quality_report.status:
            raise ValueError("Data quality checks failed.")

        logger.info("Validation completed successfully.")

    
    
    
    def _feature_engineering(self) -> None:

        date_cols = [
            c for c, t in self.df.schema.items()
            if t in (pl.Date, pl.Datetime) or c.lower() in ("date", "datetime", "timestamp")
        ]

        for col in date_cols:
            
            if self.df[col].dtype == pl.Utf8:
                self.df = self.df.with_columns(
                    pl.col(col).str.to_date(strict=False).alias(col)
                )

            
            self.df = self.df.with_columns([
                pl.col(col).dt.year().alias(f"{col}_year"),
                pl.col(col).dt.month().alias(f"{col}_month"),
                pl.col(col).dt.day().alias(f"{col}_day"),
                pl.col(col).dt.weekday().alias(f"{col}_weekday"),
                pl.col(col).dt.ordinal_day().alias(f"{col}_day_of_year"),
            ]).drop(col)  

            logger.info(f"Date column '{col}' expanded into 5 numeric features.")

      
        self.processor = FeatureProcessor(
            target_column    = self.target_column,
            problem_type     = self.problem_type,       
            scaling_strategy = ScalingStrategy.NONE,    
            leakage_threshold  = 0.995,
            variance_threshold = 0.0,
            null_threshold     = 0.90,
        )

        processed_df, metadata = self.processor.fit_transform(self.df)

        self.df                   = processed_df
        self.preprocessing_metadata = metadata

       
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

        test_size = 0.15 if len(X) < 500 else 0.2

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y if use_stratify else None
        )

        
        logger.info(f"Train-test split: {len(self.X_train)} train / {len(self.X_test)} test rows")

    
    def _train(self) -> None:
        n_rows     = len(self.X_train)
        n_features = len(self.X_train.columns)

        if n_rows < 500:
            size_tier = "small"
            overrides = {
                "n_estimators":     100,
                "max_depth":        3,
                "min_child_weight": 5,
                "reg_alpha":        0.5,
                "reg_lambda":       2.0,
                "gamma":            0.2,
            }
        elif n_rows < 5_000:
            size_tier = "medium"
            overrides = {
                "n_estimators":     300,
                "max_depth":        4,
                "min_child_weight": 3,
                "reg_alpha":        0.1,
                "reg_lambda":       1.0,
                "gamma":            0.1,
            }
        elif n_rows < 50_000:
            size_tier = "large"
            overrides = {
                "n_estimators":     500,
                "max_depth":        5,
                "min_child_weight": 1,
                "reg_alpha":        0.05,
                "reg_lambda":       1.0,
                "gamma":            0.0,
            }
        else:
            size_tier = "xlarge"
            overrides = {
                "n_estimators":     1000,
                "max_depth":        6,
                "learning_rate":    0.02,
                "min_child_weight": 1,
                "reg_alpha":        0.01,
                "reg_lambda":       1.0,
                "gamma":            0.0,
                "subsample":        0.9,
                "colsample_bytree": 0.9,
            }

        logger.info(
            f"Dataset tier: {size_tier} | "
            f"{n_rows} rows {n_features} features | "
            f"n_estimators={overrides['n_estimators']} "
            f"max_depth={overrides['max_depth']}"
        )

        
        self.model = XGBoostModel(
            problem_type=self.problem_type,
            params=overrides,
        )

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

        self.model.fit(df=train_df, target_column=self.target_column)
        self.feature_columns = self.model.feature_names
        logger.info(f"Model training completed. Features: {self.feature_columns}")

        
        
    

    

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
       
        assert hasattr(self, "processor"), (
            "Pipeline has not been trained. Call execute_pipeline() first."
        )
        assert hasattr(self, "model"), (
            "Pipeline has not been trained. Call execute_pipeline() first."
        )
 
    
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
            "drift_report": self._cached_drift,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "preprocessing_metadata": self.preprocessing_metadata,
        }
    

    def save_to_registry(self, model_name: str, registry_path: str = "ml_registry"):
       

        if not self.s3_client:
            raise ValueError("Pipeline initialized without an S3Client. Cannot register model.")
        
        normalized_model_name = model_name
        if "/" in model_name or ".parquet" in model_name:
            # Take everything after the last slash to drop folder trees
            normalized_model_name = model_name.split("/")[-1]
            # Replace a dot notation delimiter if it joins an extension tracking string
            normalized_model_name = normalized_model_name.replace(".parquet_", "_")
            normalized_model_name = normalized_model_name.replace(".parquet", "")

        logger.info(f"Dynamic name normalization: '{model_name}' ──> '{normalized_model_name}'")
        
        registry = ModelRegistry(base_path=registry_path, s3_client=self.s3_client)
        
        # Collect artifacts and metrics
        artifacts = self._collect_artifacts()
    
        raw_params = self.model.model.get_params() if hasattr(self.model.model, 'get_params') else {}

    
        sanitized_params = {
            k: (None if isinstance(v, float) and np.isnan(v) else v) 
            for k, v in raw_params.items()
        }

        next_version = registry._next_version(normalized_model_name)
       

        training_data_key = (
            f"{registry_path}/"
            f"{model_name}/"
            f"v{next_version}/"
            f"training_sample.parquet"
        )

        train_df = pl.from_pandas(self.X_train)

        train_df = train_df.with_columns(
            pl.Series(self.target_column, self.y_train)
        )

        buffer = io.BytesIO()

        train_df.write_parquet(buffer)

        buffer.seek(0)

        self.s3_client._client.put_object(
            Bucket=self.s3_client.bucket_name,
            Key=training_data_key,
            Body=buffer.getvalue(),
            ContentType="application/octet-stream",
        )
        
        
        registration_result = registry.register(
            model=self.model,
            model_name=normalized_model_name,
            metrics=self._cached_metrics,
            parameters=sanitized_params,
            problem_type=self.problem_type.name,
            training_data_key=training_data_key,
            stage="staging"
        )

        registry.promote(
            model_name=normalized_model_name,
            version=registration_result["version"],
            new_stage="production",
        )

        registration_result["stage"] = "production"


        logger.info(f"Model successfully promoted: {model_name} v{registration_result['version']}")
        return registration_result
    
    

    def _save_predictions_to_db(self, model_name: str, model_version: str) -> None:
        if self.predictions is None or self.actual_values is None:
            logger.warning("No predictions to save — skipping DB write.")
            return

        for pred, actual in zip(self.predictions, self.actual_values):
            PostgresClient.execute(
                """
                INSERT INTO model_evaluations
                    (model_name, model_version, problem_type, predicted, actual)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (model_name, model_version, self.problem_type.name, float(pred), float(actual))
            )

        logger.info(f"Saved {len(self.predictions)} predictions to model_evaluations for {model_name} v{model_version}")