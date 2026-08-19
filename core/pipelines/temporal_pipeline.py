from __future__ import annotations

import logging
import uuid
import warnings
from typing import Any, Dict, List, Optional
import time
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
from core.pipelines.base_pipeline import BasePipeline, PipelineResult, PipelineMetadata
from core.contracts.problem_type import ProblemType
from core.feature_engineering.data_quality import DataQualityAnalyzer
from core.contracts.schema_validator import SchemaValidator, DatasetSchema, ColumnContract
from core.contracts.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from core.feature_engineering.temporal_features import TemporalFeatureEngineer, TemporalFeatureConfig, TemporalResolution
from core.feature_engineering.preprocessing_utils import FeatureProcessor, ImputationStrategy
from core.models.prophet_model import ProphetModel, ForecastResult
from core.evaluation.forecasting_metrics import ForecastingMetrics
from core.evaluation.backtesting import TimeSeriesBacktester
from core.drift.drift_detector import DriftDetector
from core.models.model_registry import ModelRegistry
from storage.s3_client import S3Client
from storage.postgres_client import PostgresClient

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class TemporalPipelineConfig:
    """
    Centralised knob-board for the entire pipeline.
    All tuneable parameters live here to keep the pipeline class clean.
    """

    def __init__(
        self,
        
        forecast_horizon: int = 30,
        forecast_freq: str = "D",
        temporal_resolution: TemporalResolution = TemporalResolution.LOW,
        add_cyclic_signals: bool = True,

        # Preprocessing
        numeric_imputation: ImputationStrategy = ImputationStrategy.MEDIAN,
        leakage_threshold: float = 0.995,

        # Prophet hyperparameters
        seasonality_mode: str = "multiplicative",
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        changepoint_prior_scale: float = 0.05,
        country_holidays: Optional[str] = None,
        extra_regressors: Optional[List[str]] = None,

        # Backtesting
        run_backtest: bool = True,
        backtest_initial_train_size: Optional[int] = None,   # None → 70 % of data
        backtest_step: int = 7,

        # Drift detection
        drift_threshold: float = 0.05,

        # Circuit breaker
        min_quality_score: float = 0.40,
        max_drift_score: float = 0.30,

        # Registry
        registry_path: str = "ml_registry",
        auto_register: bool = True,
    ):
        self.forecast_horizon            = forecast_horizon
        self.forecast_freq               = forecast_freq
        self.temporal_resolution         = temporal_resolution
        self.add_cyclic_signals          = add_cyclic_signals
        self.numeric_imputation          = numeric_imputation
        self.leakage_threshold           = leakage_threshold
        self.seasonality_mode            = seasonality_mode
        self.yearly_seasonality          = yearly_seasonality
        self.weekly_seasonality          = weekly_seasonality
        self.daily_seasonality           = daily_seasonality
        self.changepoint_prior_scale     = changepoint_prior_scale
        self.country_holidays            = country_holidays
        self.extra_regressors            = extra_regressors or []
        self.run_backtest                = run_backtest
        self.backtest_initial_train_size = backtest_initial_train_size
        self.backtest_step               = backtest_step
        self.drift_threshold             = drift_threshold
        self.min_quality_score           = min_quality_score
        self.max_drift_score             = max_drift_score
        self.registry_path               = registry_path
        self.is_fitted = False
        self.auto_register               = auto_register




class TemporalPipeline(BasePipeline):
    
    def run(self) -> PipelineResult:

        start_ts = datetime.utcnow()
        start_wall = time.perf_counter()

        self._detect_problem_type()
        self._validate()
        self._feature_engineering()
        self._split()
        self._train()
        self.is_fitted = True
        if self.config.run_backtest:
            self._run_backtesting()

        metrics = self._evaluate_final()

        if self.config.auto_register:
            self._register_model()

        model_name    = self._registry_metadata.get("model_name", f"prophet_{self.experiment_id}")
        model_version = str(self._registry_metadata.get("version", "1"))
        self._save_predictions_to_db(model_name, model_version)   

        

        end_ts = datetime.utcnow()    
        duration = time.perf_counter() - start_wall
        version_raw = self._registry_metadata.get("version", "1")
        semver = f"{version_raw}.0.0" if "." not in str(version_raw) else str(version_raw)

        metadata = PipelineMetadata(
            execution_id=self.experiment_id,
            start_time=start_ts,
            end_time=end_ts,
            duration=duration,
            step_timings={}, 
            system_info={"executor": "TemporalPipeline"}
        )

        

        return PipelineResult(
            model_name=self._registry_metadata.get("model_name", f"prophet_{self.experiment_id}"),
            problem_type=self.problem_type,
            metrics=metrics,
            artifacts={
            "artifacts_key": self._registry_metadata.get("artifact_key", self.config.registry_path),
            "feature_columns": self.features,
            "model_version": semver,
            "backtest_summary": self._backtest_summary,
            "predictions": self._prediction_results
            },

            metadata=metadata
        
        )

    def __init__(
        self,
        dataframe:       pl.DataFrame,
        target_column:   str,
        s3_client: S3Client,
        datetime_column: str,                            
        experiment_id:   str = "temporal_exp",
        config:          Optional[TemporalPipelineConfig] = None,
        reference_df:    Optional[pl.DataFrame] = None, 
        
    ):
        super().__init__(
            dataframe       = dataframe,
            target_column   = target_column,
            datetime_column = datetime_column,
            experiment_id   = experiment_id,
        )

        if not datetime_column:
            raise ValueError("TemporalPipeline requires a datetime_column.")

        self.config       = config or TemporalPipelineConfig()
        self.reference_df = reference_df

        # Runtime state populated during execution 
        self._train_df: Optional[pl.DataFrame] = None
        self._test_df:  Optional[pl.DataFrame] = None
        self._forecast_result: Optional[ForecastResult] = None
        self._backtest_summary: Dict[str, Any] = {}
        self._drift_report: Any = None
        self._registry_metadata: Dict[str, Any] = {}
        self._prediction_results = []
        
        self._registry = ModelRegistry(
        base_path=self.config.registry_path,
        s3_client=s3_client  
        )

        # Sub-system instances 
        self._circuit_breaker = CircuitBreaker(
            CircuitBreakerConfig(
                min_quality_score = self.config.min_quality_score,
                max_drift_score   = self.config.max_drift_score,
            )
        )
        self._feature_engineer = TemporalFeatureEngineer(
            TemporalFeatureConfig(
                resolution        = self.config.temporal_resolution,
                add_is_weekend    = True,
                add_cyclic_signals= self.config.add_cyclic_signals,
            )
        )
        self._preprocessor = FeatureProcessor(
            target_column     = self.target_column,
            problem_type      = ProblemType.FORECASTING,
            numeric_strategy  = self.config.numeric_imputation,
            leakage_threshold = self.config.leakage_threshold,
            scaling_strategy  = "none",
        )
        self._metrics_engine = ForecastingMetrics(seasonal_period=7)
        self._drift_detector = DriftDetector(threshold=self.config.drift_threshold)


    
    def _validate(self) -> None:
        logger.info("[Validate] Running DataQualityAnalyzer …")

        # 1a. Quality analysis
        analyzer = DataQualityAnalyzer(self.df, target_column=self.target_column)
        quality_report = analyzer.analyze()

        logger.info(
            "[Validate] Quality score=%.3f  status=%s  issues=%d  warnings=%d",
            quality_report.quality_score,
            quality_report.status,
            len(quality_report.issues),
            len(quality_report.warnings),
        )

        if quality_report.issues:
            for issue in quality_report.issues:
                logger.error("[Validate] ISSUE: %s", issue)

        if quality_report.warnings:
            for warn in quality_report.warnings:
                logger.warning("[Validate] WARNING: %s", warn)

        self._circuit_breaker.check_data_quality(quality_report.quality_score)

        
        column_contracts = {
            col: ColumnContract(name=col, dtype=str(self.df.schema[col]), nullable=True)
            for col in self.df.columns
        }
        dataset_schema = DatasetSchema(
            columns       = column_contracts,
            target_column = self.target_column,
            dataset_name  = self.experiment_id,
        )
        validator = SchemaValidator(
            df           = self.df,
            schema       = dataset_schema,
            problem_type = ProblemType.FORECASTING,
        )
        validation_result = validator.validate(raise_on_failure=True)
        logger.info(
            "[Validate] Schema validation passed=%s  errors=%s",
            validation_result.valid,
            validation_result.errors,
        )

        #  Drift detection (only when reference data is provided)
        if self.reference_df is not None:
            logger.info("[Validate] Running DriftDetector …")
            numeric_features = [
                col for col, dtype in self.df.schema.items()
                if dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
                and col != self.target_column
            ]
            self._drift_report = self._drift_detector.check_drift(
                reference_df = self.reference_df,
                current_df   = self.df,
                features     = numeric_features,
            )
            logger.info(
                "[Validate] Drift detected=%s  flagged=%s",
                self._drift_report.is_drifted,
                self._drift_report.flagged_features,
            )
            if self._drift_report.is_drifted:
                avg_drift = float(
                    1.0 - np.mean(list(self._drift_report.drift_scores.values()))
                )
                self._circuit_breaker.check_drift(avg_drift)

    

    def _detect_problem_type(self) -> None:
        self.problem_type = ProblemType.FORECASTING
        logger.info("[ProblemType] Set to FORECASTING")


    def _feature_engineering(self) -> None:
        logger.info("[FeatureEngineering] Casting datetime column …")

        
        if self.df[self.datetime_column].dtype not in (pl.Date, pl.Datetime):
            self.df = self.df.with_columns(
                pl.col(self.datetime_column).str.to_datetime(strict=False)
            )

        
        self.df = self.df.sort(self.datetime_column)

        # Temporal feature decomposition
        logger.info("[FeatureEngineering] Applying TemporalFeatureEngineer …")
        self.df = self._feature_engineer.transform(
            df       = self.df,
            date_cols= [self.datetime_column],
        )

        cols_to_process = [
        col for col in self.df.columns 
        if col != self.datetime_column
        ]

        # General preprocessing (imputation, encoding, leakage removal)
        logger.info("[FeatureEngineering] Applying FeatureProcessor …")
        processed_features, meta = self._preprocessor.fit_transform(self.df.select(cols_to_process))
        self.df = pl.concat([
        self.df.select(self.datetime_column),
        processed_features
        ], how="horizontal")

        logger.info(
            "[FeatureEngineering] Preprocessing complete — "
            "dropped=%s  leaked=%s  dupes_removed=%d",
            meta.dropped_features,
            meta.leaked_features,
            meta.duplicates_removed,
        )

        # Record feature names (exclude datetime + target)
        self.features = [
            col for col in self.df.columns
            if col not in (self.target_column, self.datetime_column)
        ]
        logger.info("[FeatureEngineering] %d features produced.", len(self.features))

   

    def _split(self) -> None:
        n = self.df.height
        split_idx = max(1, n - self.config.forecast_horizon)

        self._train_df = self.df[:split_idx]
        self._test_df  = self.df[split_idx:]

        logger.info(
            "[Split] Train=%d rows  Test=%d rows  (horizon=%d)",
            self._train_df.height,
            self._test_df.height,
            self.config.forecast_horizon,
        )

   
    def _train(self) -> None:
        logger.info("[Train] Initialising ProphetModel …")

        self.model = ProphetModel(
            time_column             = self.datetime_column,
            target_column           = self.target_column,
            seasonality_mode        = self.config.seasonality_mode,
            yearly_seasonality      = self.config.yearly_seasonality,
            weekly_seasonality      = self.config.weekly_seasonality,
            daily_seasonality       = self.config.daily_seasonality,
            interval_width          = 0.95,
            country_holidays        = self.config.country_holidays,
            extra_regressors        = self.config.extra_regressors,
            changepoint_prior_scale = self.config.changepoint_prior_scale,
        )

        missing_regressors = [
            c for c in self.config.extra_regressors if c not in self._train_df.columns
        ]
        if missing_regressors:
            raise ValueError(
                f"extra_regressors {missing_regressors} not found in training data "
                f"for experiment '{self.experiment_id}'."
            )

        train_pd = self._train_df.select(
            [self.datetime_column, self.target_column] + self.config.extra_regressors
        ).to_pandas()

        self.model.fit(train_pd)
        logger.info("[Train] ProphetModel fitted successfully.")

       
   

    def _evaluate_final(self) -> Dict[str, float]:
        logger.info("[Evaluate] Generating forecast on held-out test window …")

        test_pd = self._test_df.select(
            [self.datetime_column] + self.config.extra_regressors
        ).to_pandas()

        if self.datetime_column in test_pd.columns and "ds" not in test_pd.columns:
            test_pd = test_pd.rename(columns={self.datetime_column: "ds"})

        self._forecast_result = self.model.predict(test_pd)

        y_true = self._test_df[self.target_column].to_numpy()
        y_pred = self._forecast_result.yhat[: len(y_true)]
        y_train = self._train_df[self.target_column].to_numpy()

        dates = self._test_df[self.datetime_column].to_list()



        self._prediction_results = [
            {
                "date": str(date),
                "actual": float(actual),
                "predicted": float(predicted)
            }

            for date, actual, predicted in zip(dates, y_true, y_pred)
        ]


        metric_result = self._metrics_engine.evaluate(
            y_true  = y_true,
            y_pred  = y_pred,
            y_train = y_train,
        )

        metrics = metric_result.to_dict()

        # Merge backtest summary if available
        if self._backtest_summary:
            metrics.update({
                f"bt_{k}": v for k, v in self._backtest_summary.items()
            })

        logger.info(
            "[Evaluate] MAE=%.4f  RMSE=%.4f  SMAPE=%.4f  WAPE=%.4f    MASE=%s",
            metrics.get("mae",   float("nan")),
            metrics.get("rmse",  float("nan")),
            metrics.get("smape", float("nan")),
            metrics.get("wape",  float("nan")),
            metrics.get("mase"),
        )

        return metrics

    

    def _collect_artifacts(self) -> Dict[str, Any]:
        artifacts: Dict[str, Any] = {
            "model":             self.model,
            "feature_names":     self.features,
            "forecast_result":   self._forecast_result,
            "train_rows":        self._train_df.height if self._train_df is not None else 0,
            "test_rows":         self._test_df.height  if self._test_df  is not None else 0,
        }

        if self._backtest_summary:
            artifacts["backtest_summary"] = self._backtest_summary

        if self._drift_report is not None:
            artifacts["drift_report"] = self._drift_report

        if self._registry_metadata:
            artifacts["registry_metadata"] = self._registry_metadata

        return artifacts

    

    def _run_backtesting(self) -> None:
        """Walk-forward validation with both expanding and sliding strategies."""
        logger.info("[Backtest] Starting walk-forward validation …")

        n = self._train_df.height
        initial_size = self.config.backtest_initial_train_size or max(
            self.config.forecast_horizon + 1,
            int(n * 0.70),
        )

        if initial_size >= n:
            logger.warning(
                "[Backtest] Insufficient training data for backtesting "
                "(initial_size=%d >= n=%d). Skipping.",
                initial_size, n,
            )
            return

        backtester = TimeSeriesBacktester(
            df               = self._train_df,
            datetime_column  = self.datetime_column,
            target_column    = self.target_column,
            forecast_horizon = self.config.forecast_horizon,
        )

        cfg = self.config

        def model_factory():
            """Returns a fresh ProphetModel — must be pickleable."""
            return ProphetModel(
                time_column             = self.datetime_column,
                target_column           = self.target_column,
                seasonality_mode        = cfg.seasonality_mode,
                yearly_seasonality      = cfg.yearly_seasonality,
                weekly_seasonality      = cfg.weekly_seasonality,
                daily_seasonality       = cfg.daily_seasonality,
                changepoint_prior_scale = cfg.changepoint_prior_scale,
            )

        try:
            expanding_result = backtester.run_backtest(
                strategy           = "expanding",
                model_factory      = model_factory,
                initial_train_size = initial_size,
                step               = self.config.backtest_step,
            )

            self._backtest_summary = {
                "expanding_avg_mae":  expanding_result.avg_mae,
                "expanding_avg_rmse": expanding_result.avg_rmse,
                "expanding_mae_std":  expanding_result.volatility_mae,
                "n_folds":            len(expanding_result.fold_metrics),
            }

            logger.info(
                "[Backtest] Expanding — MAE=%.4f  RMSE=%.4f  Folds=%d",
                expanding_result.avg_mae,
                expanding_result.avg_rmse,
                len(expanding_result.fold_metrics),
            )

        except Exception as exc:
            logger.warning("[Backtest] Walk-forward skipped: %s", exc)

    def _register_model(self) -> None:
        """Persists the fitted model and its metadata to the model registry."""
        try:
            self._registry_metadata = self._registry.register(
                model        = self.model,
                model_name   = f"prophet_{self.experiment_id}",
                metrics      = self._backtest_summary,
                parameters   = {
                    "seasonality_mode":        self.config.seasonality_mode,
                    "changepoint_prior_scale": self.config.changepoint_prior_scale,
                    "forecast_horizon":        self.config.forecast_horizon,
                    "extra_regressors":        self.config.extra_regressors,
                },
                problem_type = ProblemType.FORECASTING.value,
                stage        = "staging",
            )
            logger.info(
                "[Registry] Model registered — name=%s  version=%s",
                self._registry_metadata.get("model_name"),
                self._registry_metadata.get("version"),
            )
        except Exception as exc:
            logger.warning("[Registry] Registration failed (non-fatal): %s", exc)

   

    def forecast_future(self, periods: Optional[int] = None) -> ForecastResult:
        """
        Generates a future forecast beyond the training window.

        Must be called AFTER pipeline.run().

        Args:
            periods: Number of future steps. Defaults to config.forecast_horizon.

        Returns:
            ForecastResult with yhat, yhat_lower, yhat_upper arrays.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be run before calling forecast_future().")

        periods = periods or self.config.forecast_horizon
        future_df = self.model.make_future_dataframe(
            periods         = periods,
            freq            = self.config.forecast_freq,
            include_history = False,
        )
        return self.model.predict(future_df)

    def get_forecast_dataframe(self, periods: Optional[int] = None) -> pd.DataFrame:
        """
        Returns a tidy pandas DataFrame of future forecasts.

        Columns: ds | yhat | yhat_lower | yhat_upper
        """
        result = self.forecast_future(periods)
        return result.forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

    def promote_to_production(self) -> None:
        """
        Promotes the latest registered model version to the 'production' stage.
        """
        if not self._registry_metadata:
            raise RuntimeError("No model has been registered. Ensure auto_register=True.")

        self._registry.promote(
            model_name = self._registry_metadata["model_name"],
            version    = self._registry_metadata["version"],
            new_stage  = "production",
        )
        logger.info(
            "[Registry] Model '%s' v%s promoted to production.",
            self._registry_metadata["model_name"],
            self._registry_metadata["version"],
        )

    def _save_predictions_to_db(self, model_name: str, model_version: str) -> None:
        if not self._prediction_results:
            logger.warning("[DB] No predictions to save — skipping DB write.")
            return

        for row in self._prediction_results:
            PostgresClient.execute(
                """
                INSERT INTO forecast_evaluations
                    (model_name, model_version, problem_type, predicted, actual, prediction_date)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    model_name,
                    model_version,
                    ProblemType.FORECASTING.name,
                    float(row["predicted"]),
                    float(row["actual"]),
                    row.get("date"),
                )
            )

        logger.info(
            "[DB] Saved %d predictions to forecast_evaluations for %s v%s",
            len(self._prediction_results),
            model_name,
            model_version,
        )



def train_all_products(
    full_df: pl.DataFrame,
    product_col: str,
    target_column: str,
    datetime_column: str,
    s3_client: S3Client,
    config: Optional[TemporalPipelineConfig] = None,
) -> Dict[str, PipelineResult]:
    results = {}
    for product_id in full_df[product_col].unique().to_list():
        product_df = full_df.filter(pl.col(product_col) == product_id)

        if product_df.height < 30:  # guard: not enough history to forecast
            logger.warning("Skipping product %s — insufficient rows (%d)", product_id, product_df.height)
            continue

        pipeline = TemporalPipeline(
            dataframe       = product_df,
            target_column   = target_column,
            datetime_column = datetime_column,
            s3_client       = s3_client,
            experiment_id   = f"product_{product_id}",
            config          = config,
        )
        results[str(product_id)] = pipeline.run()
    return results