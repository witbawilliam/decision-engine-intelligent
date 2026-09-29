from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import joblib
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

logger = logging.getLogger(__name__)

PROPHET_DS_COL    = "ds"
PROPHET_Y_COL     = "y"
DEFAULT_INTERVAL  = 0.95
MIN_TRAINING_ROWS = 2


@dataclass(frozen=True)
class ForecastResult:
    forecast_df:    pd.DataFrame
    yhat:           np.ndarray
    yhat_lower:     np.ndarray
    yhat_upper:     np.ndarray
    horizon_rows:   int
    model_id:       str
    regressor_cols: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id":       self.model_id,
            "horizon_rows":   self.horizon_rows,
            "regressor_cols": self.regressor_cols,
            "yhat":           self.yhat.tolist(),
            "yhat_lower":     self.yhat_lower.tolist(),
            "yhat_upper":     self.yhat_upper.tolist(),
        }


class ProphetModel:
   
    

    def __init__(
        self,
        time_column:             str   = PROPHET_DS_COL,
        target_column:           str   = PROPHET_Y_COL,
        seasonality_mode:        str   = "multiplicative",
        yearly_seasonality:      bool  = True,
        weekly_seasonality:      bool  = True,
        daily_seasonality:       bool  = False,
        interval_width:          float = DEFAULT_INTERVAL,
        country_holidays:        Optional[str]       = None,
        extra_regressors:        Optional[List[str]] = None,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
    ) -> None:
        self.time_column             = time_column
        self.target_column           = target_column
        self.seasonality_mode        = seasonality_mode
        self.yearly_seasonality      = yearly_seasonality
        self.weekly_seasonality      = weekly_seasonality
        self.daily_seasonality       = daily_seasonality
        self.interval_width          = interval_width
        self.country_holidays        = country_holidays
        self.extra_regressors        = extra_regressors or []
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale

        self._model:     Optional[Prophet]      = None
        self._train_df:  Optional[pd.DataFrame] = None
        self._is_fitted: bool                   = False
        self.model_id:   str                    = str(uuid.uuid4())

        logger.info(
            "[ProphetModel:%s] Initialised — time_col='%s', target_col='%s', "
            "mode='%s', regressors=%s",
            self.model_id, self.time_column, self.target_column,
            self.seasonality_mode, self.extra_regressors,
        )


    def _to_prophet_df(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in [self.time_column, self.target_column] if c not in df.columns]
        if missing:
            raise KeyError(f"[ProphetModel:{self.model_id}] Missing columns: {missing}")

        prophet_df = df[[self.time_column, self.target_column]].copy()
        prophet_df = prophet_df.rename(
            columns={self.time_column: PROPHET_DS_COL, self.target_column: PROPHET_Y_COL}
        )
        try:
            prophet_df[PROPHET_DS_COL] = pd.to_datetime(prophet_df[PROPHET_DS_COL])
        except Exception as exc:
            raise ValueError(f"Cannot parse '{self.time_column}' as datetime: {exc}") from exc

        try:
            prophet_df[PROPHET_Y_COL] = prophet_df[PROPHET_Y_COL].astype(float)
        except Exception as exc:
            raise ValueError(f"Cannot cast '{self.target_column}' to float: {exc}") from exc

        for reg in self.extra_regressors:
            if reg not in df.columns:
                raise KeyError(f"Regressor '{reg}' not in columns: {list(df.columns)}")
            prophet_df[reg] = df[reg].values

        return prophet_df
    

    def _build_prophet(self) -> Prophet:
        model = Prophet(
            seasonality_mode        = self.seasonality_mode,
            yearly_seasonality      = self.yearly_seasonality,
            weekly_seasonality      = self.weekly_seasonality,
            daily_seasonality       = self.daily_seasonality,
            interval_width          = self.interval_width,
            changepoint_prior_scale = self.changepoint_prior_scale,
            seasonality_prior_scale = self.seasonality_prior_scale,
        )
        if self.country_holidays:
            model.add_country_holidays(country_name=self.country_holidays)
        for reg in self.extra_regressors:
            model.add_regressor(reg)
        return model


    def fit(self, df: pd.DataFrame) -> "ProphetModel":
        logger.info("[ProphetModel:%s] fit() — input shape: %s", self.model_id, df.shape)
        prophet_df = self._to_prophet_df(df)

        null_count = prophet_df[PROPHET_Y_COL].isna().sum()
        if null_count > 0:
            logger.warning("[ProphetModel:%s] Dropping %d NaN rows.", self.model_id, null_count)
            prophet_df = prophet_df.dropna(subset=[PROPHET_Y_COL])

        
        if len(prophet_df) < MIN_TRAINING_ROWS:
            raise ValueError(
                f"Insufficient data: requres at least {MIN_TRAINING_ROWS} rows after cleaning."
            )
            

        self._model     = self._build_prophet()
        self._model.fit(prophet_df)
        self._train_df  = prophet_df
        self._is_fitted = True

        logger.info("[ProphetModel:%s] Training complete — %d rows.", self.model_id, len(prophet_df))
        return self
    

    def make_future_dataframe(
        self, periods: int, freq: str = "D", include_history: bool = False
    ) -> pd.DataFrame:
        self._assert_fitted("make_future_dataframe")
        return self._model.make_future_dataframe(
            periods=periods, freq=freq, include_history=include_history
        )


    def predict(self, future_df: pd.DataFrame) -> ForecastResult:
        self._assert_fitted("predict")

        if future_df.empty:
            raise ValueError(f"[ProphetModel:{self.model_id}] future_df is empty.")

        if self.time_column in future_df.columns and PROPHET_DS_COL not in future_df.columns:
            future_df = future_df.rename(columns={self.time_column: PROPHET_DS_COL})

        if PROPHET_DS_COL not in future_df.columns:
            raise ValueError(
                f"[ProphetModel:{self.model_id}] future_df must contain 'ds'. "
                f"Found: {list(future_df.columns)}"
            )

        for reg in self.extra_regressors:
            if reg not in future_df.columns:
                raise KeyError(f"Regressor '{reg}' missing from future_df at predict time.")
            
        logger.info(
            "[ProphetModel:%s] future_df nulls:\n%s",
            self.model_id,
            future_df.isnull().sum()
        )

        forecast = self._model.predict(future_df)
        logger.info("[ProphetModel:%s] Prediction — %d rows.", self.model_id, len(forecast))

        return ForecastResult(
            forecast_df    = forecast,
            yhat           = forecast["yhat"].to_numpy(),
            yhat_lower     = forecast["yhat_lower"].to_numpy(),
            yhat_upper     = forecast["yhat_upper"].to_numpy(),
            horizon_rows   = len(future_df),
            model_id       = self.model_id,
            regressor_cols = list(self.extra_regressors),
        )
    

    def cross_validate(
        self,
        horizon:  str           = "30 days",
        period:   str           = "15 days",
        initial:  Optional[str] = None,
        parallel: Optional[str] = None,
    ) -> pd.DataFrame:
        self._assert_fitted("cross_validate")
        kwargs: Dict[str, Any] = dict(model=self._model, horizon=horizon, period=period)
        if initial:  kwargs["initial"]  = initial
        if parallel: kwargs["parallel"] = parallel
        logger.info("[ProphetModel:%s] CV — horizon='%s'.", self.model_id, horizon)
        return cross_validation(**kwargs)
    

    def cross_validation_metrics(
        self,
        horizon: str = "30 days",
        period:  str = "15 days",
        initial: Optional[str] = None,
    ) -> Dict[str, float]:
        cv_df = self.cross_validate(horizon=horizon, period=period, initial=initial)
        perf  = performance_metrics(cv_df)
        summary = {
            "rmse": float(perf["rmse"].mean()),
            "mae":  float(perf["mae"].mean()),
            "mape": float(perf["mape"].mean()),
        }
        logger.info(
            "[ProphetModel:%s] CV metrics — RMSE=%.4f MAE=%.4f MAPE=%.4f",
            self.model_id, summary["rmse"], summary["mae"], summary["mape"],
        )
        return summary
    

    def save(self, path: str) -> str:
        self._assert_fitted("save")
        abs_path = str(Path(path).resolve())
        os.makedirs(Path(abs_path).parent, exist_ok=True)
        joblib.dump(self, abs_path)
        logger.info("[ProphetModel:%s] Saved to '%s'.", self.model_id, abs_path)
        return abs_path

    @classmethod
    def load(cls, path: str) -> "ProphetModel":
        abs_path = str(Path(path).resolve())
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Checkpoint not found at '{abs_path}'.")
        instance = joblib.load(abs_path)
        logger.info("[ProphetModel:%s] Loaded from '%s'.", instance.model_id, abs_path)
        return instance
    

    def plot_components(self, forecast_df: pd.DataFrame) -> None:
        """Render component plots. Only use in notebooks, never in API workers."""
        self._assert_fitted("plot_components")
        self._model.plot_components(forecast_df)


    def _assert_fitted(self, caller: str) -> None:
        if not self._is_fitted or self._model is None:
            raise RuntimeError(
                f"[ProphetModel:{self.model_id}] '{caller}' called before fit()."
            )


    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"ProphetModel(id={self.model_id[:8]}, status={status}, "
            f"mode={self.seasonality_mode}, regressors={self.extra_regressors})"
        )