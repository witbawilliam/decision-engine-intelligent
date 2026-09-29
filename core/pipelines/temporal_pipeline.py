from __future__ import annotations

import logging
import time
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from core.contracts.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from core.contracts.problem_type import ProblemType
from core.contracts.schema_validator import ColumnContract, DatasetSchema, SchemaValidator
from core.drift.drift_detector import DriftDetector
from core.evaluation.backtesting import TimeSeriesBacktester
from core.evaluation.forecasting_metrics import ForecastingMetrics
from core.feature_engineering.data_quality import DataQualityAnalyzer
from core.feature_engineering.preprocessing_utils import FeatureProcessor, ImputationStrategy
from core.feature_engineering.temporal_features import TemporalFeatureConfig, TemporalFeatureEngineer, TemporalResolution
from core.models.model_registry import ModelRegistry
from core.models.xgboost_model_forecast import ForecastResult, XGBoostForecastModel
from core.pipelines.base_pipeline import BasePipeline, PipelineMetadata, PipelineResult
from storage.postgres_client import PostgresClient
from storage.s3_client import S3Client

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class TemporalPipelineConfig:
    def __init__(
        self,
        model_type: str = "xgboost",
        forecast_horizon: int = 7,
        forecast_freq: str = "D",
        temporal_resolution: TemporalResolution = TemporalResolution.LOW,
        add_cyclic_signals: bool = True,
        numeric_imputation: ImputationStrategy = ImputationStrategy.MEDIAN,
        xgb_lags: Optional[List[int]] = None,
        xgb_rolling_windows: Optional[List[int]] = None,
        xgb_params: Optional[Dict[str, Any]] = None,
        run_backtest: bool = True,
        backtest_initial_train_size: Optional[int] = None,
        backtest_step: int = 7,
        drift_threshold: float = 0.05,
        min_quality_score: float = 0.40,
        max_drift_score: float = 0.30,
        registry_path: str = "ml_registry",
        auto_register: bool = True,
    ):
        self.model_type = model_type.lower()
        self.forecast_horizon = forecast_horizon
        self.forecast_freq = forecast_freq
        self.temporal_resolution = temporal_resolution
        self.add_cyclic_signals = add_cyclic_signals
        self.numeric_imputation = numeric_imputation
        self.xgb_lags = xgb_lags or [1, 7, 14, 28]
        self.xgb_rolling_windows = xgb_rolling_windows or [7, 14, 28]
        self.xgb_params = xgb_params or {}
        self.run_backtest = run_backtest
        self.backtest_initial_train_size = backtest_initial_train_size
        self.backtest_step = backtest_step
        self.drift_threshold = drift_threshold
        self.min_quality_score = min_quality_score
        self.max_drift_score = max_drift_score
        self.registry_path = registry_path
        self.auto_register = auto_register


class TemporalPipeline(BasePipeline):
    def __init__(
        self,
        dataframe: pl.DataFrame,
        target_column: str,
        s3_client: S3Client,
        datetime_column: str,
        product_column: Optional[str] = None,
        experiment_id: str = "temporal_exp",
        config: Optional[TemporalPipelineConfig] = None,
        reference_df: Optional[pl.DataFrame] = None,
    ):
        super().__init__(
            dataframe=dataframe,
            target_column=target_column,
            datetime_column=datetime_column,
            experiment_id=experiment_id,
        )

        if not datetime_column:
            raise ValueError("TemporalPipeline requires a datetime_column.")

        self.config = config or TemporalPipelineConfig()
        self.reference_df = reference_df
        self.product_column = product_column

        if self.product_column is not None and self.product_column not in dataframe.columns:
            raise ValueError(f"product_column '{self.product_column}' not found in dataframe.")

        self._train_df: Optional[pl.DataFrame] = None
        self._test_df: Optional[pl.DataFrame] = None
        self._forecast_result: Optional[ForecastResult] = None
        self._backtest_summary: Dict[str, Any] = {}
        self._drift_report: Any = None
        self._registry_metadata: Dict[str, Any] = {}
        self._prediction_results: List[Dict[str, Any]] = []
        self._final_metrics: Dict[str, float] = {}

        self._registry = ModelRegistry(base_path=self.config.registry_path, s3_client=s3_client)
        self._circuit_breaker = CircuitBreaker(
            CircuitBreakerConfig(
                min_quality_score=self.config.min_quality_score,
                max_drift_score=self.config.max_drift_score,
            )
        )
        self._feature_engineer = TemporalFeatureEngineer(
            TemporalFeatureConfig(
                resolution=self.config.temporal_resolution,
                add_is_weekend=True,
                add_cyclic_signals=self.config.add_cyclic_signals,
            )
        )

        protected_columns = [self.datetime_column, self.target_column]
        if self.product_column:
            protected_columns.append(self.product_column)

        self._preprocessor = FeatureProcessor(
            target_column=self.target_column,
            problem_type=ProblemType.FORECASTING,
            numeric_strategy=self.config.numeric_imputation,
            protected_columns=protected_columns,
        )
        self._metrics_engine = ForecastingMetrics(seasonal_period=7)
        self._drift_detector = DriftDetector(threshold=self.config.drift_threshold)

    def run(self) -> PipelineResult:
        start_ts = datetime.utcnow()
        start_wall = time.perf_counter()

        self._detect_problem_type()
        self._validate()
        # Ensure temporal split happens before feature transformations
        self._split()
        self._feature_engineering()
        self._train()
        self.is_fitted = True

        if self.config.run_backtest:
            self._run_backtesting()

        metrics = self._evaluate_final()
        self._final_metrics = metrics

        if self.config.auto_register:
            self._register_model()

        model_name = self._registry_metadata.get("model_name", f"xgboost_{self.experiment_id}")
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
            system_info={"executor": "TemporalPipeline"},
        )

        return PipelineResult(
            model_name=model_name,
            problem_type=self.problem_type,
            metrics=metrics,
            artifacts={
                "artifacts_key": self._registry_metadata.get("artifact_key", self.config.registry_path),
                "feature_columns": self.features,
                "model_version": semver,
                "backtest_summary": self._backtest_summary,
                "predictions": self._prediction_results,
            },
            metadata=metadata,
        )

    def _detect_problem_type(self) -> None:
        self.problem_type = ProblemType.FORECASTING

    def _validate(self) -> None:
        analyzer = DataQualityAnalyzer(self.df, target_column=self.target_column)
        quality_report = analyzer.analyze()
        self._circuit_breaker.check_data_quality(quality_report.quality_score)

        column_contracts = {
            col: ColumnContract(name=col, dtype=str(self.df.schema[col]), nullable=True)
            for col in self.df.columns
        }
        dataset_schema = DatasetSchema(
            columns=column_contracts,
            target_column=self.target_column,
            dataset_name=self.experiment_id,
        )
        validator = SchemaValidator(
            df=self.df, schema=dataset_schema, problem_type=ProblemType.FORECASTING
        )
        validator.validate(raise_on_failure=True)

        if self.reference_df is not None:
            numeric_features = [
                col for col, dtype in self.df.schema.items()
                if dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
                and col != self.target_column
            ]
            self._drift_report = self._drift_detector.check_drift(
                reference_df=self.reference_df, current_df=self.df, features=numeric_features
            )
            if self._drift_report.is_drifted:
                avg_drift = float(1.0 - np.mean(list(self._drift_report.drift_scores.values())))
                self._circuit_breaker.check_drift(avg_drift)

    def _split(self) -> None:
        """Splits raw dataset causally based on temporal cutoffs per entity."""
        if self.df[self.datetime_column].dtype not in (pl.Date, pl.Datetime):
            self.df = self.df.with_columns(
                pl.col(self.datetime_column).str.to_datetime(strict=False)
            ).drop_nulls(subset=[self.datetime_column])

        sort_cols = [self.product_column, self.datetime_column] if self.product_column else [self.datetime_column]
        self.df = self.df.sort(sort_cols)

        if self.product_column is not None:
            # Date-based temporal cutoff calculation per product group
            unique_dates = self.df[self.datetime_column].unique().sort()
            total_dates = len(unique_dates)
            horizon = self.config.forecast_horizon

            if total_dates <= horizon:
                raise ValueError(
                    f"Insufficient time periods ({total_dates}) for forecast horizon ({horizon})."
                )

            split_date = unique_dates[total_dates - horizon]
            self._train_df = self.df.filter(pl.col(self.datetime_column) < split_date)
            self._test_df = self.df.filter(pl.col(self.datetime_column) >= split_date)
        else:
            split_idx = max(1, self.df.height - self.config.forecast_horizon)
            self._train_df = self.df[:split_idx]
            self._test_df = self.df[split_idx:]

    def _feature_engineering(self) -> None:
        """Applies feature transformations while preserving canonical column ordering."""
        #  Record original input column ordering
        original_cols = list(self._train_df.columns)

    
        self._train_df = self._feature_engineer.transform(df=self._train_df, date_cols=[self.datetime_column])
        self._test_df = self._feature_engineer.transform(df=self._test_df, date_cols=[self.datetime_column])

        #  Preserve tracking metadata in canonical order: [product_column, datetime_column]
        meta_cols = []
        if self.product_column and self.product_column in self._train_df.columns:
            meta_cols.append(self.product_column)
        meta_cols.append(self.datetime_column)

        cols_to_process = [col for col in self._train_df.columns if col not in meta_cols]

        # Transform feature subsets without altering tracking columns
        processed_train, _ = self._preprocessor.fit_transform(self._train_df.select(cols_to_process))
        res_test = self._preprocessor.transform(self._test_df.select(cols_to_process))
        processed_test = res_test[0] if isinstance(res_test, tuple) else res_test

        # Recombine metadata and engineered features
        train_combined = pl.concat([self._train_df.select(meta_cols), processed_train], how="horizontal")
        test_combined = pl.concat([self._test_df.select(meta_cols), processed_test], how="horizontal")

        # Re-align column order with original schema + newly generated features
        new_features = [c for c in train_combined.columns if c not in original_cols]
        ordered_cols = [c for c in original_cols if c in train_combined.columns] + new_features

        self._train_df = train_combined.select(ordered_cols)
        self._test_df = test_combined.select(ordered_cols)

        # Update active feature list
        exclude_cols = set(meta_cols) | {self.target_column}
        self.features = [col for col in self._train_df.columns if col not in exclude_cols]

    def _train(self) -> None:
        if self.config.model_type == "xgboost":
            self.model = XGBoostForecastModel(
                target_column=self.target_column,
                time_column=self.datetime_column,
                product_column=self.product_column,
                lags=self.config.xgb_lags,
                rolling_windows=self.config.xgb_rolling_windows,
                params=self.config.xgb_params,
                model_name=f"xgb_global_{self.experiment_id}",
            )
            self.model.fit(self._train_df)

    def _run_backtesting(self) -> None:
        
        if self.product_column is not None:
            n = self._train_df[self.datetime_column].n_unique()
        else:
            n = self._train_df.height

        initial_size = self.config.backtest_initial_train_size or max(
            self.config.forecast_horizon + 1, int(n * 0.70)
        )
        if initial_size >= n:
            logger.warning(
                "[Backtest] Skipped — initial_size (%d) >= available %s (%d). "
                "Not enough history for even one walk-forward fold.",
                initial_size,
                "unique dates" if self.product_column is not None else "rows",
                n,
            )
            return

        backtester = TimeSeriesBacktester(
            df=self._train_df,
            datetime_column=self.datetime_column,
            target_column=self.target_column,
            forecast_horizon=self.config.forecast_horizon,
            product_column=self.product_column,
        )
        cfg = self.config

        def model_factory():
            return XGBoostForecastModel(
                target_column=self.target_column,
                time_column=self.datetime_column,
                product_column=self.product_column,
                lags=cfg.xgb_lags,
                rolling_windows=cfg.xgb_rolling_windows,
                params=cfg.xgb_params,
            )

        try:
            expanding_result = backtester.run_backtest(
                strategy="expanding",
                model_factory=model_factory,
                initial_train_size=initial_size,
                step=self.config.backtest_step,
            )
            self._backtest_summary = {
                "expanding_avg_mae": expanding_result.avg_mae,
                "expanding_avg_rmse": expanding_result.avg_rmse,
                "expanding_mae_std": expanding_result.volatility_mae,
                "n_folds": len(expanding_result.fold_metrics),
            }
           
            logger.info(
                "[Backtest] Expanding walk-forward complete — %d folds, "
                "avg_mae=%.3f, avg_rmse=%.3f, mae_std=%.3f",
                len(expanding_result.fold_metrics),
                expanding_result.avg_mae,
                expanding_result.avg_rmse,
                expanding_result.volatility_mae,
            )
        except Exception as exc:
            logger.warning("[Backtest] Walk-forward skipped: %s", exc)

    def _evaluate_final(self) -> Dict[str, float]:
        self._forecast_result = self.model.predict(periods=self.config.forecast_horizon)
        pred_df = self._forecast_result.to_polars()

        rename_map = {"date": self.datetime_column}
        if self.product_column is not None:
            rename_map["product"] = self.product_column
        pred_df = pred_df.rename(rename_map)

        if self.product_column is not None:
            joined = self._test_df.join(
                pred_df, on=[self.product_column, self.datetime_column], how="inner"
            )
        else:
            joined = self._test_df.join(

                pred_df, on=self.datetime_column, how="inner"
            )

        logger.info(
            "[Evaluate] Join result: %d rows (test_df had %d, forecast had %d)",
            joined.height, self._test_df.height, pred_df.height,
        )

        if self.product_column is not None and joined.height > 0:
            for product_id in joined[self.product_column].unique().sort().to_list():
                sub = joined.filter(pl.col(self.product_column) == product_id)
                sub_true = sub[self.target_column].to_numpy()
                sub_pred = sub["yhat"].to_numpy()

                if len(sub_true) == 0:
                    continue

                abs_actual_sum = float(np.abs(sub_true).sum())
                if abs_actual_sum == 0:
                    logger.warning(
                        "[Evaluate][%s] rows=%d — WAPE undefined (sum of actuals is 0).",
                        product_id, len(sub_true),
                    )
                    continue

                product_wape = float(np.abs(sub_true - sub_pred).sum() / abs_actual_sum * 100)
                logger.info(
                    "[Evaluate][%s] rows=%d  WAPE=%.2f  sum_actual=%.2f  sum_pred=%.2f",
                    product_id, len(sub_true), product_wape,
                    float(sub_true.sum()), float(sub_pred.sum()),
                )

       
        if joined.height > 0:
            forecast_start = pred_df[self.datetime_column].min()
            joined_with_step = joined.with_columns(
                ((pl.col(self.datetime_column) - forecast_start).dt.total_days() + 1)
                .alias("_days_ahead")
            )

            for days_ahead in joined_with_step["_days_ahead"].unique().sort().to_list():
                step_sub = joined_with_step.filter(pl.col("_days_ahead") == days_ahead)
                step_true = step_sub[self.target_column].to_numpy()
                step_pred = step_sub["yhat"].to_numpy()

                if len(step_true) == 0:
                    continue

                step_mae = float(np.mean(np.abs(step_true - step_pred)))
                abs_actual_sum = float(np.abs(step_true).sum())
                step_wape = (
                    float(np.abs(step_true - step_pred).sum() / abs_actual_sum * 100)
                    if abs_actual_sum > 0
                    else None
                )
                logger.info(
                    "[Evaluate][day_ahead=%d] rows=%d  MAE=%.3f  WAPE=%s",
                    int(days_ahead), len(step_true), step_mae,
                    f"{step_wape:.2f}" if step_wape is not None else "undefined (sum_actual=0)",
                )

        y_true = joined[self.target_column].to_numpy()
        y_pred = joined["yhat"].to_numpy()
        y_train = self._train_df[self.target_column].to_numpy()
        dates = joined[self.datetime_column].to_list()

        self._prediction_results = [
            {"date": str(d), "actual": float(a), "predicted": float(p)}
            for d, a, p in zip(dates, y_true, y_pred)
        ]

       
        metric_result = self._metrics_engine.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            y_train=y_train,
            y_train_groups=(
                self._train_df[self.product_column].to_numpy()
                if self.product_column is not None
                else None
            ),
        )
        metrics = metric_result.to_dict()
        if self._backtest_summary:
            metrics.update({f"bt_{k}": v for k, v in self._backtest_summary.items()})

       
        logger.info("[Evaluate] Full metrics: %s", metrics)

        return metrics

    

    def _register_model(self) -> None:
        try:
            self._registry_metadata = self._registry.register(
                model=self.model,
                model_name=f"xgboost_{self.experiment_id}",
                metrics={**self._backtest_summary, **self._final_metrics},
                parameters={
                    "forecast_horizon": self.config.forecast_horizon,
                    "lags": self.config.xgb_lags,
                    "rolling_windows": self.config.xgb_rolling_windows,
                    "xgb_params": self.config.xgb_params,
                    "training_scope": "all_products_global",
                },
                problem_type=ProblemType.FORECASTING.value,
                stage="staging",
                preprocessor=self._preprocessor,
            )
        except Exception as exc:
            logger.warning("[Registry] Registration failed: %s", exc)


    def promote_to_production(self, metric_name: str = "mase", max_threshold: float = 0.80) -> bool:
        """Evaluates final performance metrics against quality thresholds and updates registry stage."""
        current_metric = self._final_metrics.get(metric_name)
        if current_metric is None:
            logger.warning("[Promotion] Metric '%s' not found in evaluation metrics.", metric_name)
            return False

        if current_metric <= max_threshold:
            model_name = self._registry_metadata.get("model_name", f"xgboost_{self.experiment_id}")
            version = self._registry_metadata.get("version", "1")

           
            self._registry.promote(
                model_name=model_name,
                version=version,
                new_stage="production",
            )
            logger.info(
                "[Promotion] Model passed metric quality gate (%s = %.2f <= %.2f). Promoted to production.",
                metric_name, current_metric, max_threshold
            )
            return True
        else:
            logger.warning(
                "[Promotion] Model failed metric gate (%s = %.2f > %.2f). Promotion rejected.",
                metric_name, current_metric, max_threshold
            )
            return False

    def forecast_future(self, periods: Optional[int] = None) -> ForecastResult:
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be run before calling forecast_future().")
        periods = periods if periods is not None else self.config.forecast_horizon
        return self.model.predict(periods=periods)

    def _save_predictions_to_db(self, model_name: str, model_version: str) -> None:
        if not self._prediction_results:
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
                ),
            )


def train_all_products(
    full_df: pl.DataFrame,
    product_col: str,
    target_column: str,
    datetime_column: str,
    s3_client: S3Client,
    config: Optional[TemporalPipelineConfig] = None,
) -> Dict[str, Any]:
    """Preprocesses a multi-product panel dataset and trains a global XGBoost forecaster."""
    config = config or TemporalPipelineConfig()

    required_columns = [product_col, target_column, datetime_column]
    missing_columns = [col for col in required_columns if col not in full_df.columns]
    if missing_columns:
        raise KeyError(
            f"Missing required column(s): {missing_columns}. "
            f"Available columns: {full_df.columns}"
        )

    panel_df = (
        full_df.select(
            [
                pl.col(datetime_column),
                pl.col(product_col),
                pl.col(target_column).cast(pl.Float64, strict=False).alias(target_column),
            ]
        )
        .drop_nulls(subset=[datetime_column, product_col, target_column])
    )

    if panel_df.schema[datetime_column] not in (pl.Date, pl.Datetime):
        panel_df = panel_df.with_columns(
            pl.col(datetime_column).str.to_datetime(strict=False).alias(datetime_column)
        ).drop_nulls(subset=[datetime_column])

    panel_df = (
        panel_df.group_by([product_col, datetime_column])
        .agg(pl.col(target_column).sum().alias(target_column))
        .sort([product_col, datetime_column])
    )

    total_products = panel_df[product_col].n_unique()

    if total_products < 1:
        raise ValueError("No valid products found after preprocessing.")

    if panel_df.height < 30:
        raise ValueError(
            f"Insufficient total training rows. Got {panel_df.height}, need at least 30."
        )

    logger.info(
        "[GlobalTraining] Preparing ONE global model across %d products and %d observations.",
        total_products,
        panel_df.height,
    )

    pipeline = TemporalPipeline(
        dataframe=panel_df,
        target_column=target_column,
        datetime_column=datetime_column,
        product_column=product_col,
        s3_client=s3_client,
        experiment_id="all_products_global_xgboost",
        config=config,
    )

    result = pipeline.run()

    try:
        pipeline.promote_to_production(
            metric_name="mase",
            max_threshold=0.90,
        )
    except Exception as exc:
        logger.warning(
            "[GlobalTraining] Model trained successfully but promotion was skipped: %s",
            exc,
        )

    return {
        "global": {
            "status": "success",
            "result": result,
            "training_scope": "global_panel",
            "model_type": "xgboost",
            "models_trained": 1,
            "products_included": total_products,
            "training_rows": panel_df.height,
            "product_column": product_col,
            "target_column": target_column,
            "datetime_column": datetime_column,
        }
    }