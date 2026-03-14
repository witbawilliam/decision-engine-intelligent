from __future__ import annotations

import logging
from typing import Dict, Any

import polars as pl

from core.pipelines.base_pipeline import BasePipeline
from core.contracts.schema_inference import SchemaInference
from core.contracts.schema_validator import SchemaValidator
from core.feature_engineering.data_quality import DataQualityAnalyzer
from core.feature_engineering.preprocessing_utils import FeatureProcessor
from core.labeling.target_identifier import TargetIdentifier
from core.models.xgboost_model import XGBoostModel
from core.evaluation.regression_metrics import RegressionMetrics
from core.evaluation.feature_importance import XGBExplainer
from core.drift.drift_detector import DriftDetector
from core.contracts.problem_type import ProblemType
from sklearn.preprocessing import LabelEncoder
from core.models.model_registry import ModelRegistry
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional

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


class TabularPipeline(BasePipeline):
    """
    Enterprise-Grade Tabular ML Pipeline
    Fully modular, contract-driven, and production-aligned.
    """

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
            
            # Evaluation
            final_metrics = self._evaluate_final()
            
            # Persistence
            # Assuming your save_to_registry returns the registration info
            registration = self.save_to_registry(model_name=model_name)

            # Construct the Contract
            result = PipelineResult(
                status="SUCCESS",
                model_name=model_name,
                model_version=str(registration.get("version", "unknown")),
                metrics=final_metrics,
                artifacts_path=registration.get("path", ""),
                feature_columns=self.feature_columns,
                metadata={
                    "problem_type": self.problem_type.name,
                    "is_drifted": self._check_drift()["is_drifted"]
                }
            )
            
            logger.info(f"Pipeline finished successfully for {model_name}")
            return result

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise e

    
    

    def _detect_problem_type(self) -> None:
        identifier = TargetIdentifier(
            df=self.df,
            force_target=self.target_column
        )
        detection_result = identifier.identify()

        detected_type = detection_result.problem_type


        #Normalize external enum to platform enum

        if detected_type.name in ["BINARY", "MULTICLASS", "CLASSIFICATION"]:

            self.problem_type = ProblemType.CLASSIFICATION

        else:

           self.problem_type = detected_type



        logger.info(f"Detected problem type: {self.problem_type.name}")



    def _validate(self) -> None:
        """
        Validation flow:
        1. Detect problem type
        2. Infer schema
        3. Validate schema contract
        4. Data quality audit
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
        Stateless feature processing.
        Returns processed dataframe + metadata.
        """

        processor = FeatureProcessor(
            target_column=self.target_column
        )

        processed_df, metadata = processor.process(self.df)

        self.df = processed_df
        self.preprocessing_metadata = metadata

        # Derive feature lists from processed dataframe
        self.numeric_features = [
            c for c, t in self.df.schema.items()
            if t.is_numeric() and c != self.target_column
        ]

        self.categorical_features = [
            c for c, t in self.df.schema.items()
            if not t.is_numeric() and c != self.target_column
        ]

        logger.info(
            f"Feature engineering complete: "
            f"{len(self.numeric_features)} numeric, "
            f"{len(self.categorical_features)} categorical"
        )

    
    
    

    def _split(self) -> None:
        from sklearn.model_selection import train_test_split

        X = self.df.drop(self.target_column).to_pandas()
        y = self.df[self.target_column].to_pandas()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y if self.problem_type == ProblemType.CLASSIFICATION else None,
        )

        
        # self.feature_columns = list(self.X_train.columns)

        logger.info("Train-test split completed.")

    
    
    

    def _train(self) -> None:
        self.model = XGBoostModel(problem_type=self.problem_type)

        y_train = self.y_train

        if self.problem_type == ProblemType.CLASSIFICATION:
            self.label_encoder = LabelEncoder()
            y_train = self.label_encoder.fit_transform(self.y_train)

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
        test_df = pl.from_pandas(self.X_test)
        test_df = test_df.with_columns(
            pl.Series(self.target_column, self.y_test)
        )

        result = self.model.predict(test_df.drop(self.target_column))
        predictions = result.predictions

        if self.problem_type == ProblemType.REGRESSION:
                evaluator = RegressionMetrics()
                metric_result = evaluator.evaluate(
                y_true=self.y_test,
                y_pred=predictions,
                n_features=len(self.feature_columns)
            )

                return metric_result.to_dict()

        
     
        
        if self.problem_type == ProblemType.CLASSIFICATION:
        
            from sklearn.metrics import accuracy_score, f1_score, precision_score

            # Ensure y_true matches the format of predictions
            y_true = self.y_test
            if hasattr(self, 'label_encoder'):
                y_true = self.label_encoder.transform(self.y_test)

            metrics = {
                "accuracy": float(accuracy_score(y_true, predictions)),
                "f1_weighted": float(f1_score(y_true, predictions, average="weighted")),
                "precision": float(precision_score(y_true, predictions, average="weighted"))
            }
            logger.info(f"Classification Metrics: {metrics}")
            return metrics
    
        return {}
    


    def predict(self, data: Any) -> Any:
        """
        High-level inference entry point.
        Resolves the 'AttributeError: DataFrame object has no attribute select'.
        """
        # Ensure input data is converted to Polars if it arrives as Pandas/Dict
        if not isinstance(data, pl.DataFrame):
            # This check prevents the 'no attribute select' error in the inference test
            inference_df = pl.DataFrame(data) if not isinstance(data, pd.DataFrame) else pl.from_pandas(data)
        else:
            inference_df = data

        # If your FeatureProcessor logic is required for raw data at inference:
        # Note: You should typically save the processor or logic for production
        processed_inference_df, _ = FeatureProcessor(
            target_column=self.target_column
        ).process(inference_df)

        return self.model.predict(processed_inference_df.drop(self.target_column, strict=False))
        
    
    
    def _post_training_analysis(self) -> Dict[str, Any]:
        analyzer = XGBExplainer(
            self.model.model,   
            feature_names=self.feature_columns
        )

        # Replace compute() with the actual method defined inside XGBExplainer
        importance_report = analyzer.explain(self.X_train)

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
        # 1. Initialize Registry
        registry = ModelRegistry(base_path=registry_path)
        
        # 2. Collect artifacts and metrics
        artifacts = self._collect_artifacts()
        final_metrics = self._evaluate_final()
        
        
        registration_result = registry.register(
            model=self.model,
            model_name=model_name,
            metrics=final_metrics,
            parameters=self.model.model.get_params() if hasattr(self.model.model, 'get_params') else {},
            problem_type=self.problem_type.name,
            stage="staging"
        )

        logger.info(f"Model successfully registered: {model_name} v{registration_result['version']}")
        return registration_result