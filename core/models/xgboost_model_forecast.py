from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import polars as pl
from xgboost import XGBRegressor


logger = logging.getLogger(__name__)


_MIN_SCALE_FACTOR = 1e-3


_GLOBAL_SCALE_KEY = "__global__"


class NotFittedError(RuntimeError):
    """Raised when the forecasting model is used before fitting."""
    pass


@dataclass(frozen=True)
class ForecastResult:
    """Encapsulates output metrics and predictions for panel forecasts."""

    dates: List[Any]
    products: List[Any]
    yhat: np.ndarray
    horizon: int
    model_id: str
    feature_importance: Dict[str, float] = field(default_factory=dict)
    inference_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dates": [str(d) for d in self.dates],
            "products": [str(p) for p in self.products],
            "yhat": self.yhat.tolist(),
            "horizon": self.horizon,
            "model_id": self.model_id,
            "feature_importance": self.feature_importance,
            "inference_time_ms": self.inference_time_ms,
        }

    def to_polars(self) -> pl.DataFrame:
        """Convert forecast output to a structured Polars DataFrame."""
        return pl.DataFrame(
            {
                "date": self.dates,
                "product": self.products,
                "yhat": self.yhat.tolist(),
            }
        )


class XGBoostForecastModel:
    

    def __init__(
        self,
        target_column: str,
        time_column: str = "ds",
        product_column: Optional[str] = None,
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        freq_days: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        model_name: str = "xgboost_global_forecaster",
    ) -> None:
        self.target_column = target_column
        self.time_column = time_column
        self.product_column = product_column
        self.lags = sorted(set(lags or [1, 7, 14, 28]))
        self.rolling_windows = sorted(set(rolling_windows or [7, 14, 28]))
        self.freq_days = freq_days
        self.model_name = model_name

        if not self.lags:
            raise ValueError("At least one lag must be configured.")
        if any(lag < 1 for lag in self.lags):
            raise ValueError("All lag values must be >= 1.")
        if any(w < 1 for w in self.rolling_windows):
            raise ValueError("All rolling window sizes must be >= 1.")
        if self.freq_days is not None and self.freq_days < 1:
            raise ValueError("freq_days must be >= 1.")

        self._max_lag = max(self.lags)
        self._max_rolling_window = max(self.rolling_windows) if self.rolling_windows else 1
        self._min_history = max(self._max_lag, self._max_rolling_window)

        default_params: Dict[str, Any] = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
            "objective": "reg:squarederror",
            "min_child_weight": 5,
            "enable_categorical": True,
        }
        default_params.update(params or {})

        self.model = XGBRegressor(**default_params)
        self.feature_names: Optional[List[str]] = None
        self._is_fitted = False
        self._history: Optional[pl.DataFrame] = None
        self._freq_days: int = 1
        self._products: List[Any] = []

        
        self._scale_factors: Dict[str, float] = {}

    def _validate_input_schema(self, df: pl.DataFrame) -> None:
        if not isinstance(df, pl.DataFrame):
            raise TypeError("Input must be a Polars DataFrame.")

        required_columns = [self.time_column, self.target_column]
        if self.product_column is not None:
            required_columns.append(self.product_column)

        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise KeyError(f"Missing required schema columns: {missing}")

        time_dtype = df.schema[self.time_column]
        if time_dtype not in (pl.Date, pl.Datetime):
            raise TypeError(
                f"time_column '{self.time_column}' must be Date or Datetime, got {time_dtype}."
            )

        if self.product_column is not None and df[self.product_column].null_count() > 0:
            raise ValueError(f"Product column '{self.product_column}' contains null entries.")

    def _validate_duplicates(self, df: pl.DataFrame) -> None:
        if self.product_column is not None:
            unique_rows = df.select([self.product_column, self.time_column]).unique().height
            duplicates = df.height - unique_rows
            if duplicates > 0:
                raise ValueError(
                    f"Detected {duplicates} duplicate entity-timestamp rows in panel."
                )
        else:
            if df[self.time_column].n_unique() != df.height:
                raise ValueError(f"Duplicate timestamps detected in '{self.time_column}'.")

    def _compute_scale_factors(self, clean_df: pl.DataFrame) -> Dict[str, float]:
        """
        computes one scale factor per entity — the mean of that entity's
        historical target values — used to normalize the target before
        training. A floor (_MIN_SCALE_FACTOR) prevents division by zero or
        near-zero for an extremely sparse or all-zero series. Mean (not
        median) is used deliberately: intermittent low-count demand series
        can have a median of 0, which would make the scale factor useless.
        """
        if self.product_column is not None:
            stats = (
                clean_df.group_by(self.product_column)
                .agg(pl.col(self.target_column).mean().alias("_scale"))
            )
            return {
                str(row[self.product_column]): max(float(row["_scale"] or 0.0), _MIN_SCALE_FACTOR)
                for row in stats.iter_rows(named=True)
            }
        else:
            overall_mean = float(clean_df[self.target_column].mean() or 0.0)
            return {_GLOBAL_SCALE_KEY: max(overall_mean, _MIN_SCALE_FACTOR)}

    def _scale_expr(self) -> pl.Expr:
        """
        returns a Polars expression producing the per-row scale factor,
        looked up by product_column via a fixed mapping. Used to both
        normalize (divide) and, via its reciprocal, is not needed for
        denormalization — that happens on the numpy array in
        _denormalize_series, since predict() collects results as plain
        Python dicts, not a DataFrame, at that stage.
        """
        if self.product_column is not None:
            return (
                pl.col(self.product_column)
                .cast(pl.Utf8)
                .replace(self._scale_factors, default=_MIN_SCALE_FACTOR)
                .cast(pl.Float64)
            )
        return pl.lit(self._scale_factors[_GLOBAL_SCALE_KEY])

    def _normalize_target(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        divides the target column by each row's entity scale factor.
        Called once in fit(), on clean_df, before feature generation and
        before self._history is sliced — meaning _build_features,
        _prepare_prediction_row, and _forecast_product all operate purely on
        normalized values without needing any awareness that normalization
        exists. This containment is deliberate: those three methods have
        already required several rounds of bug fixes, and adding scale
        handling inside them would multiply that risk.
        """
        return df.with_columns(
            (pl.col(self.target_column) / self._scale_expr()).alias(self.target_column)
        )

    def _denormalize_series(self, values: np.ndarray, products: List[Any]) -> np.ndarray:
        """
        the single point where normalized predictions are converted back
        to real units, at the end of predict(). `products` must be aligned
        1:1 with `values` (same order, same length).
        """
        if self.product_column is not None:
            scales = np.array(
                [self._scale_factors.get(str(p), 1.0) for p in products],
                dtype=float,
            )
        else:
            scales = np.full(len(values), self._scale_factors[_GLOBAL_SCALE_KEY])
        return values * scales

    def _build_features(self, df: pl.DataFrame, for_training: bool) -> pl.DataFrame:
        """Construct causal multi-level lag, rolling, and calendar features via Polars."""
        out = df.sort([self.product_column, self.time_column]) if self.product_column else df.sort(self.time_column)
        exprs = []

        # Entity-isolated Lags
        for lag in self.lags:
            expr = pl.col(self.target_column).shift(lag)
            if self.product_column is not None:
                expr = expr.over(self.product_column)
            exprs.append(expr.alias(f"{self.target_column}_lag_{lag}"))

        # Entity-isolated Causal Rolling Means (min_periods=window prevents train/serve skew)
        for window in self.rolling_windows:
            expr = (
                pl.col(self.target_column)
                .shift(1)
                .rolling_mean(window_size=window, min_periods=window)
            )
            if self.product_column is not None:
                expr = expr.over(self.product_column)
            exprs.append(expr.alias(f"{self.target_column}_rollmean_{window}"))

        out = out.with_columns(exprs)

        # Calendar Temporal Features
        out = out.with_columns(
            [
                pl.col(self.time_column).dt.year().alias("_cal_year"),
                pl.col(self.time_column).dt.month().alias("_cal_month"),
                pl.col(self.time_column).dt.day().alias("_cal_day"),
                pl.col(self.time_column).dt.weekday().alias("_cal_weekday"),
                pl.col(self.time_column).dt.ordinal_day().alias("_cal_day_of_year"),
            ]
        )

        if for_training:
            feature_cols = [c for c in out.columns if c not in (self.time_column, self.target_column)]
            numeric_cols = [c for c in feature_cols if c != self.product_column]
            if numeric_cols:
                out = out.drop_nulls(subset=numeric_cols)

        return out

    def _infer_freq_days(self, df: pl.DataFrame) -> int:
        dates = df.select(self.time_column).unique().sort(self.time_column)
        if dates.height < 2:
            return 1

        diffs = (
            dates.with_columns(
                pl.col(self.time_column).diff().dt.total_days().alias("_diff")
            )
            .drop_nulls("_diff")["_diff"]
        )

        if diffs.len() == 0:
            return 1

        mode = diffs.mode()
        return max(int(mode[0]), 1) if mode.len() > 0 else 1

    def _validate_product_history(self, df: pl.DataFrame) -> None:
        if self.product_column is None:
            if df.height <= self._min_history:
                raise ValueError(
                    f"Insufficient historical points. Need > {self._min_history}, got {df.height}."
                )
            return

        counts = df.group_by(self.product_column).agg(pl.len().alias("_count"))
        insufficient = counts.filter(pl.col("_count") <= self._min_history)

        if insufficient.height > 0:
            bad_entities = insufficient[self.product_column].head(10).to_list()
            raise ValueError(
                f"{insufficient.height} entity level(s) have insufficient history "
                f"(<= {self._min_history} rows). Examples: {bad_entities}"
            )

    def fit(self, df: pl.DataFrame) -> XGBoostForecastModel:
        """Fit global panel XGBoost forecaster across multi-level entities."""
        start_time = time.perf_counter()
        self._validate_input_schema(df)

        cols = [self.time_column, self.target_column]
        if self.product_column is not None:
            cols.insert(1, self.product_column)

        clean_df = (
            df.select(cols)
            .with_columns(pl.col(self.target_column).cast(pl.Float64))
            .drop_nulls(subset=[self.target_column])
        )

        if clean_df.height == 0:
            raise ValueError("Zero rows remain after removing null target values.")


        if self.product_column is not None:
            clean_df = clean_df.with_columns(pl.col(self.product_column).cast(pl.Utf8))
            clean_df = clean_df.sort([self.product_column, self.time_column])
        else:
            clean_df = clean_df.sort(self.time_column)

        self._validate_duplicates(clean_df)
        self._validate_product_history(clean_df)

        if self.product_column is not None:
            self._products = (
                clean_df.select(self.product_column)
                .unique()
                .sort(self.product_column)[self.product_column]
                .to_list()
            )

       
        self._scale_factors = self._compute_scale_factors(clean_df)
        clean_df = self._normalize_target(clean_df)

        # Explicit configuration preferred over automatic inference
        if self.freq_days is not None:
            self._freq_days = self.freq_days
        else:
            self._freq_days = self._infer_freq_days(clean_df)
            logger.info("Inferred forecast cadence: %d day(s).", self._freq_days)

        featured = self._build_features(clean_df, for_training=True)

        if featured.height == 0:
            raise ValueError("Feature generation yielded zero valid training records.")

        logger.info(
            "[XGBoostForecastModel:%s] Post-feature-generation training rows: %d (raw input: %d)",
            self.model_name, featured.height, clean_df.height,
        )

        self.feature_names = [
            c for c in featured.columns if c not in (self.time_column, self.target_column)
        ]

        X = featured.select(self.feature_names).to_pandas()
        if self.product_column is not None:
            X[self.product_column] = pd.Categorical(
                X[self.product_column], categories=self._products
            )

        y = featured[self.target_column].to_numpy()
        self.model.fit(X, y)

        
        history_rows = self._max_lag + self._max_rolling_window + 2
        if self.product_column is not None:
            self._history = (
                clean_df.sort([self.product_column, self.time_column])
                .group_by(self.product_column, maintain_order=True)
                .tail(history_rows)
                .sort([self.product_column, self.time_column])
            )
        else:
            self._history = clean_df.sort(self.time_column).tail(history_rows)

        self._is_fitted = True
        logger.info(
            "[XGBoostForecastModel:%s] Fitted panel model in %.2fs.",
            self.model_name,
            time.perf_counter() - start_time,
        )
        return self

    def _prepare_prediction_row(
        self, history: pl.DataFrame, next_date: Any, product: Any = None
    ) -> pl.DataFrame:
        #  operates purely on whatever units `history`'s target
        # column is in (normalized), with no awareness of scaling.
        time_dtype = history.schema[self.time_column]
        target_dtype = history.schema[self.target_column]

        data: Dict[str, List[Any]] = {
            self.time_column: [next_date],
            self.target_column: [None],
        }
        if self.product_column is not None:
            data[self.product_column] = [product]

        placeholder = pl.DataFrame(data).with_columns(
            [
                pl.col(self.time_column).cast(time_dtype),
                pl.col(self.target_column).cast(target_dtype),
            ]
        )

        placeholder = placeholder.select(history.columns)
        candidate = pl.concat([history, placeholder], how="vertical")
        featured = self._build_features(candidate, for_training=False)
        return featured.filter(pl.col(self.time_column) == next_date).tail(1)

    def _forecast_product(
        self, product_history: pl.DataFrame, periods: int, product: Any
    ) -> List[Dict[str, Any]]:
       
        working = product_history.clone()
        results: List[Dict[str, Any]] = []

        time_dtype = working.schema[self.time_column]
        target_dtype = working.schema[self.target_column]

        for _ in range(periods):
            last_date = working[self.time_column][-1]
            next_date = last_date + timedelta(days=self._freq_days)

            row = self._prepare_prediction_row(
                history=working, next_date=next_date, product=product
            )

            X = row.select(self.feature_names).to_pandas()
            if self.product_column is not None:
                X[self.product_column] = pd.Categorical(
                    X[self.product_column], categories=self._products
                )

            yhat = float(self.model.predict(X)[0])
            results.append({"date": next_date, "product": product, "yhat": yhat})

            observed_data: Dict[str, List[Any]] = {}
            if self.product_column is not None:
                observed_data[self.product_column] = [product]
            observed_data[self.target_column] = [yhat]
            observed_data[self.time_column] = [next_date]

            observed = pl.DataFrame(observed_data).with_columns(
                [
                    pl.col(self.time_column).cast(time_dtype),
                    pl.col(self.target_column).cast(target_dtype),
                ]
            )

            observed = observed.select(working.columns)
            keep_rows = self._max_lag + self._max_rolling_window + 2
            working = pl.concat([working, observed], how="vertical").sort(self.time_column).tail(keep_rows)

        return results

    def predict(
        self, periods: int, products: Optional[List[Any]] = None
    ) -> ForecastResult:
        """Autoregressively forecast multi-level entities for N step periods into the future."""
        if not self._is_fitted:
            raise NotFittedError("Call fit() before invoking predict().")
        if periods < 1:
            raise ValueError("periods parameter must be >= 1.")
        if self._history is None:
            raise RuntimeError("Model historical context is uninitialized.")

        start_time = time.perf_counter()

        if self.product_column is None:
            rows = self._forecast_product(
                product_history=self._history, periods=periods, product=None
            )
        else:
            requested_products = products if products is not None else self._products
            unknowns = [p for p in requested_products if p not in self._products]
            if unknowns:
                raise ValueError(f"Cannot forecast unobserved entity levels: {unknowns}")

            rows = []
            for product in requested_products:
                product_history = self._history.filter(
                    pl.col(self.product_column) == product
                ).sort(self.time_column)

                product_rows = self._forecast_product(
                    product_history=product_history, periods=periods, product=product
                )
                rows.extend(product_rows)

        inference_time_ms = (time.perf_counter() - start_time) * 1000

        
        raw_yhat = np.array([r["yhat"] for r in rows], dtype=float)
        products_list = [r["product"] for r in rows]
        denormalized_yhat = self._denormalize_series(raw_yhat, products_list)

        return ForecastResult(
            dates=[r["date"] for r in rows],
            products=products_list,
            yhat=denormalized_yhat,
            horizon=periods,
            model_id=self.model_name,
            feature_importance=self._get_feature_importance(),
            inference_time_ms=inference_time_ms,
        )

    def _get_feature_importance(self) -> Dict[str, float]:
        if not self._is_fitted or self.feature_names is None:
            return {}
        try:
            booster = self.model.get_booster()
            scores = booster.get_score(importance_type="gain")
            importance = {
                name: float(scores.get(name, scores.get(f"f{idx}", 0.0)) or 0.0)
                for idx, name in enumerate(self.feature_names)
            }
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        except Exception as exc:
            logger.warning("Could not extract feature importance: %s", exc)
            return {}

    def save(self, path: Union[str, Path]) -> None:
        """Serialize model artifacts, feature state, and trailing context to disk."""
        if not self._is_fitted:
            raise NotFittedError("Cannot save an unfitted model.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "target_column": self.target_column,
            "time_column": self.time_column,
            "product_column": self.product_column,
            "lags": self.lags,
            "rolling_windows": self.rolling_windows,
            "freq_days": self.freq_days,
            "model_name": self.model_name,
            "feature_names": self.feature_names,
            "_is_fitted": self._is_fitted,
            "_freq_days": self._freq_days,
            "_products": self._products,
            "_scale_factors": self._scale_factors, 
           
            "_history": self._history.to_arrow() if self._history is not None else None,
            "model": self.model,
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> XGBoostForecastModel:
        """Deserialize disk artifact into a ready-to-infer XGBoostForecastModel."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file path does not exist: {path}")

        state = joblib.load(path)
        instance = cls(
            target_column=state["target_column"],
            time_column=state["time_column"],
            product_column=state["product_column"],
            lags=state["lags"],
            rolling_windows=state["rolling_windows"],
            freq_days=state.get("freq_days"),
            model_name=state["model_name"],
        )
        instance.feature_names = state["feature_names"]
        instance._is_fitted = state["_is_fitted"]
        instance._freq_days = state["_freq_days"]
        instance._products = state["_products"]
      
        instance._scale_factors = state.get("_scale_factors", {})
        if state["_history"] is not None:
            instance._history = pl.from_arrow(state["_history"])
        instance.model = state["model"]

        return instance