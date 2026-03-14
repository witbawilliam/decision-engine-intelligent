import os
import uuid
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from core.models.prophet_model import (
    ProphetModel,
    ForecastResult,
    PROPHET_DS_COL,
    PROPHET_Y_COL,
    MIN_TRAINING_ROWS,
)



def make_daily_df(n: int = 365, seed: int = 0) -> pd.DataFrame:
    """Clean synthetic daily sales series."""
    np.random.seed(seed)
    dates  = pd.date_range("2022-01-01", periods=n, freq="D")
    values = 100 + np.arange(n) * 0.05 + np.random.normal(0, 5, n)
    return pd.DataFrame({"ds": dates, "y": values})


def make_custom_col_df(n: int = 100) -> pd.DataFrame:
    """Same data but with caller-chosen column names."""
    dates  = pd.date_range("2023-01-01", periods=n, freq="D")
    values = np.random.rand(n) * 200
    return pd.DataFrame({"date": dates, "revenue": values})


def make_regressor_df(n: int = 200) -> pd.DataFrame:
    """DataFrame with an extra regressor column."""
    df       = make_daily_df(n)
    df["promo"] = np.random.randint(0, 2, n).astype(float)
    return df


@pytest.fixture
def daily_df():
    return make_daily_df()


@pytest.fixture
def custom_df():
    return make_custom_col_df()


@pytest.fixture
def regressor_df():
    return make_regressor_df()


@pytest.fixture
def fitted_model(daily_df):
    """A ProphetModel that has been fit() — mocked so tests run fast."""
    model = ProphetModel()
    mock_prophet = MagicMock()
    with patch.object(model, "_build_prophet", return_value=mock_prophet):
        model._model     = mock_prophet
        model._train_df  = daily_df.rename(columns={"ds": PROPHET_DS_COL, "y": PROPHET_Y_COL})
        model._is_fitted = True
    return model



def _make_forecast_df(n: int = 10) -> pd.DataFrame:
    """Minimal Prophet-style forecast output."""
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "ds":         dates,
        "yhat":       np.ones(n) * 110,
        "yhat_lower": np.ones(n) * 100,
        "yhat_upper": np.ones(n) * 120,
        "trend":      np.ones(n) * 105,
    })



class TestInit:

    def test_default_model_type_is_prophet(self):
        m = ProphetModel()
        assert m.seasonality_mode == "multiplicative"

    def test_default_columns_are_ds_and_y(self):
        m = ProphetModel()
        assert m.time_column   == "ds"
        assert m.target_column == "y"

    def test_custom_columns_stored(self):
        m = ProphetModel(time_column="date", target_column="revenue")
        assert m.time_column   == "date"
        assert m.target_column == "revenue"

    def test_model_id_is_uuid_string(self):
        m = ProphetModel()
        uuid.UUID(m.model_id)  # raises if invalid

    def test_each_instance_has_unique_model_id(self):
        ids = {ProphetModel().model_id for _ in range(10)}
        assert len(ids) == 10

    def test_model_is_none_before_fit(self):
        assert ProphetModel()._model is None

    def test_is_fitted_false_before_fit(self):
        assert ProphetModel()._is_fitted is False

    def test_metrics_empty_before_fit(self):
        assert ProphetModel().extra_regressors == []

    def test_extra_regressors_stored(self):
        m = ProphetModel(extra_regressors=["promo", "price"])
        assert m.extra_regressors == ["promo", "price"]

    def test_none_extra_regressors_becomes_empty_list(self):
        m = ProphetModel(extra_regressors=None)
        assert m.extra_regressors == []

    def test_interval_width_stored(self):
        m = ProphetModel(interval_width=0.80)
        assert m.interval_width == 0.80

    def test_country_holidays_stored(self):
        m = ProphetModel(country_holidays="US")
        assert m.country_holidays == "US"

    def test_changepoint_prior_scale_stored(self):
        m = ProphetModel(changepoint_prior_scale=0.1)
        assert m.changepoint_prior_scale == 0.1



class TestToProphetDf:

    def test_renames_ds_and_y_columns(self, daily_df):
        m  = ProphetModel()
        df = m._to_prophet_df(daily_df)
        assert PROPHET_DS_COL in df.columns
        assert PROPHET_Y_COL  in df.columns

    def test_renames_custom_columns(self, custom_df):
        m  = ProphetModel(time_column="date", target_column="revenue")
        df = m._to_prophet_df(custom_df)
        assert PROPHET_DS_COL in df.columns
        assert PROPHET_Y_COL  in df.columns

    def test_ds_is_datetime(self, daily_df):
        m  = ProphetModel()
        df = m._to_prophet_df(daily_df)
        assert pd.api.types.is_datetime64_any_dtype(df[PROPHET_DS_COL])

    def test_y_is_float(self, daily_df):
        m  = ProphetModel()
        df = m._to_prophet_df(daily_df)
        assert df[PROPHET_Y_COL].dtype == float

    def test_missing_time_column_raises_key_error(self, daily_df):
        m = ProphetModel()
        with pytest.raises(KeyError, match="Missing columns"):
            m._to_prophet_df(daily_df.drop(columns=["ds"]))

    def test_missing_target_column_raises_key_error(self, daily_df):
        m = ProphetModel()
        with pytest.raises(KeyError, match="Missing columns"):
            m._to_prophet_df(daily_df.drop(columns=["y"]))

    def test_unparseable_datetime_raises_value_error(self):
        df = pd.DataFrame({"ds": ["not-a-date"] * 5, "y": [1.0] * 5})
        m  = ProphetModel()
        with pytest.raises(ValueError, match="Cannot parse"):
            m._to_prophet_df(df)

    def test_non_numeric_target_raises_value_error(self):
        df = pd.DataFrame({
            "ds": pd.date_range("2023-01-01", periods=5),
            "y":  ["a", "b", "c", "d", "e"],
        })
        m = ProphetModel()
        with pytest.raises(ValueError, match="Cannot cast"):
            m._to_prophet_df(df)

    def test_regressor_attached(self, regressor_df):
        m  = ProphetModel(extra_regressors=["promo"])
        df = m._to_prophet_df(regressor_df)
        assert "promo" in df.columns

    def test_missing_regressor_raises_key_error(self, daily_df):
        m = ProphetModel(extra_regressors=["missing_col"])
        with pytest.raises(KeyError, match="missing_col"):
            m._to_prophet_df(daily_df)

    def test_row_count_preserved(self, daily_df):
        m  = ProphetModel()
        df = m._to_prophet_df(daily_df)
        assert len(df) == len(daily_df)



class TestBuildProphet:

    def test_returns_prophet_instance(self):
        from prophet import Prophet
        m = ProphetModel()
        with patch("core.models.prophet_model.Prophet") as MockProphet:
            MockProphet.return_value = MagicMock(spec=Prophet)
            result = m._build_prophet()
        assert result is not None

    def test_country_holidays_added(self):
        m = ProphetModel(country_holidays="GB")
        mock_prophet = MagicMock()
        with patch("core.models.prophet_model.Prophet", return_value=mock_prophet):
            m._build_prophet()
        mock_prophet.add_country_holidays.assert_called_once_with(country_name="GB")

    def test_no_holidays_when_none(self):
        m = ProphetModel(country_holidays=None)
        mock_prophet = MagicMock()
        with patch("core.models.prophet_model.Prophet", return_value=mock_prophet):
            m._build_prophet()
        mock_prophet.add_country_holidays.assert_not_called()

    def test_regressors_registered(self):
        m = ProphetModel(extra_regressors=["promo", "price"])
        mock_prophet = MagicMock()
        with patch("core.models.prophet_model.Prophet", return_value=mock_prophet):
            m._build_prophet()
        assert mock_prophet.add_regressor.call_count == 2

    def test_seasonality_mode_passed_to_prophet(self):
        m = ProphetModel(seasonality_mode="additive")
        with patch("core.models.prophet_model.Prophet") as MockProphet:
            MockProphet.return_value = MagicMock()
            m._build_prophet()
        call_kwargs = MockProphet.call_args[1]
        assert call_kwargs["seasonality_mode"] == "additive"


class TestFit:

    def test_fit_sets_is_fitted_true(self, daily_df):
        m = ProphetModel()
        mock_prophet = MagicMock()
        with patch.object(m, "_build_prophet", return_value=mock_prophet):
            m.fit(daily_df)
        assert m._is_fitted is True

    def test_fit_returns_self_for_chaining(self, daily_df):
        m = ProphetModel()
        mock_prophet = MagicMock()
        with patch.object(m, "_build_prophet", return_value=mock_prophet):
            result = m.fit(daily_df)
        assert result is m

    def test_fit_calls_prophet_fit(self, daily_df):
        m = ProphetModel()
        mock_prophet = MagicMock()
        with patch.object(m, "_build_prophet", return_value=mock_prophet):
            m.fit(daily_df)
        mock_prophet.fit.assert_called_once()

    def test_fit_stores_train_df(self, daily_df):
        m = ProphetModel()
        mock_prophet = MagicMock()
        with patch.object(m, "_build_prophet", return_value=mock_prophet):
            m.fit(daily_df)
        assert m._train_df is not None

    def test_fit_drops_nan_target_rows(self):
        df = make_daily_df(50)
        df.loc[0:4, "y"] = np.nan   # introduce 5 NaN rows
        m = ProphetModel()
        mock_prophet = MagicMock()
        with patch.object(m, "_build_prophet", return_value=mock_prophet):
            m.fit(df)
        fit_df = mock_prophet.fit.call_args[0][0]
        assert fit_df[PROPHET_Y_COL].isna().sum() == 0

    def test_fit_raises_if_too_few_rows_after_nan_drop(self):
        df = pd.DataFrame({
            "ds": pd.date_range("2023-01-01", periods=3),
            "y":  [np.nan, np.nan, np.nan],
        })
        m = ProphetModel()
        with patch.object(m, "_build_prophet", return_value=MagicMock()):
            with pytest.raises(ValueError, match="Insufficient rows"):
                m.fit(df)

    def test_fit_raises_on_missing_column(self, daily_df):
        m = ProphetModel()
        with pytest.raises(KeyError):
            m.fit(daily_df.drop(columns=["y"]))

    def test_fit_with_custom_columns(self, custom_df):
        m = ProphetModel(time_column="date", target_column="revenue")
        mock_prophet = MagicMock()
        with patch.object(m, "_build_prophet", return_value=mock_prophet):
            m.fit(custom_df)
        assert m._is_fitted is True

    def test_fit_with_regressors(self, regressor_df):
        m = ProphetModel(extra_regressors=["promo"])
        mock_prophet = MagicMock()
        with patch.object(m, "_build_prophet", return_value=mock_prophet):
            m.fit(regressor_df)
        assert m._is_fitted is True



class TestMakeFutureDataframe:

    def test_raises_if_not_fitted(self):
        m = ProphetModel()
        with pytest.raises(RuntimeError, match="fit()"):
            m.make_future_dataframe(periods=10)

    def test_returns_dataframe(self, daily_df):
        m = ProphetModel()
        mock_prophet = MagicMock()
        mock_prophet.make_future_dataframe.return_value = pd.DataFrame(
            {"ds": pd.date_range("2023-01-01", periods=10)}
        )
        with patch.object(m, "_build_prophet", return_value=mock_prophet):
            m.fit(daily_df)
            result = m.make_future_dataframe(periods=10)
        assert isinstance(result, pd.DataFrame)

    def test_correct_periods_passed(self, daily_df):
        m = ProphetModel()
        mock_prophet = MagicMock()
        mock_prophet.make_future_dataframe.return_value = pd.DataFrame(
            {"ds": pd.date_range("2023-01-01", periods=30)}
        )
        with patch.object(m, "_build_prophet", return_value=mock_prophet):
            m.fit(daily_df)
            m.make_future_dataframe(periods=30, freq="D")
        mock_prophet.make_future_dataframe.assert_called_once_with(
            periods=30, freq="D", include_history=False
        )

    def test_include_history_passed_through(self, daily_df):
        m = ProphetModel()
        mock_prophet = MagicMock()
        mock_prophet.make_future_dataframe.return_value = pd.DataFrame(
            {"ds": pd.date_range("2023-01-01", periods=10)}
        )
        with patch.object(m, "_build_prophet", return_value=mock_prophet):
            m.fit(daily_df)
            m.make_future_dataframe(periods=10, include_history=True)
        call_kwargs = mock_prophet.make_future_dataframe.call_args[1]
        assert call_kwargs["include_history"] is True



class TestPredict:

    def _fitted_model_with_mock(self, daily_df):
        m = ProphetModel()
        mock_prophet = MagicMock()
        mock_prophet.predict.return_value = _make_forecast_df(10)
        with patch.object(m, "_build_prophet", return_value=mock_prophet):
            m.fit(daily_df)
        m._model = mock_prophet
        return m

    def test_raises_if_not_fitted(self):
        m = ProphetModel()
        future = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=5)})
        with pytest.raises(RuntimeError, match="fit()"):
            m.predict(future)

    def test_raises_on_empty_future_df(self, daily_df):
        m = self._fitted_model_with_mock(daily_df)
        with pytest.raises(ValueError, match="empty"):
            m.predict(pd.DataFrame())

    def test_raises_if_ds_column_missing(self, daily_df):
        m = self._fitted_model_with_mock(daily_df)
        bad_future = pd.DataFrame({"wrong_col": pd.date_range("2023-01-01", periods=5)})
        with pytest.raises(ValueError, match="'ds'"):
            m.predict(bad_future)

    def test_returns_forecast_result(self, daily_df):
        m = self._fitted_model_with_mock(daily_df)
        future = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=10)})
        result = m.predict(future)
        assert isinstance(result, ForecastResult)

    def test_forecast_result_yhat_shape(self, daily_df):
        m = self._fitted_model_with_mock(daily_df)
        n = 10
        future = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n)})
        result = m.predict(future)
        assert len(result.yhat) == n

    def test_forecast_result_has_lower_and_upper(self, daily_df):
        m = self._fitted_model_with_mock(daily_df)
        future = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=5)})
        result = m.predict(future)
        assert result.yhat_lower is not None
        assert result.yhat_upper is not None

    def test_horizon_rows_correct(self, daily_df):
        m = self._fitted_model_with_mock(daily_df)
        n = 7
        future = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n)})
        result = m.predict(future)
        assert result.horizon_rows == n

    def test_model_id_in_result(self, daily_df):
        m = self._fitted_model_with_mock(daily_df)
        future = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=5)})
        result = m.predict(future)
        assert result.model_id == m.model_id

    def test_caller_time_column_normalised_to_ds(self):
        """If caller uses 'date' column, it must be renamed to 'ds' internally."""
        m = ProphetModel(time_column="date", target_column="revenue")
        mock_prophet = MagicMock()
        mock_prophet.predict.return_value = _make_forecast_df(5)
        m._model     = mock_prophet
        m._is_fitted = True

        future = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=5)})
        result = m.predict(future)
        assert isinstance(result, ForecastResult)

    def test_missing_regressor_at_predict_raises(self, daily_df):
        m = ProphetModel(extra_regressors=["promo"])
        mock_prophet = MagicMock()
        mock_prophet.predict.return_value = _make_forecast_df(5)
        m._model     = mock_prophet
        m._is_fitted = True

        future = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=5)})
        # 'promo' is missing
        with pytest.raises(KeyError, match="promo"):
            m.predict(future)

    def test_regressor_cols_in_result(self, daily_df):
        m = ProphetModel(extra_regressors=["promo"])
        mock_prophet = MagicMock()
        mock_prophet.predict.return_value = _make_forecast_df(5)
        m._model     = mock_prophet
        m._is_fitted = True

        future = pd.DataFrame({
            "ds":    pd.date_range("2023-01-01", periods=5),
            "promo": [0, 1, 0, 1, 0],
        })
        result = m.predict(future)
        assert "promo" in result.regressor_cols



class TestForecastResultToDict:

    def _make_result(self, n: int = 5) -> ForecastResult:
        return ForecastResult(
            forecast_df    = _make_forecast_df(n),
            yhat           = np.ones(n) * 110,
            yhat_lower     = np.ones(n) * 100,
            yhat_upper     = np.ones(n) * 120,
            horizon_rows   = n,
            model_id       = str(uuid.uuid4()),
            regressor_cols = ["promo"],
        )

    def test_to_dict_has_required_keys(self):
        d = self._make_result().to_dict()
        for key in ("model_id", "horizon_rows", "regressor_cols", "yhat", "yhat_lower", "yhat_upper"):
            assert key in d

    def test_yhat_is_list(self):
        d = self._make_result().to_dict()
        assert isinstance(d["yhat"], list)

    def test_yhat_lower_is_list(self):
        d = self._make_result().to_dict()
        assert isinstance(d["yhat_lower"], list)

    def test_horizon_rows_correct(self):
        d = self._make_result(7).to_dict()
        assert d["horizon_rows"] == 7

    def test_regressor_cols_passed_through(self):
        d = self._make_result().to_dict()
        assert d["regressor_cols"] == ["promo"]



class TestCrossValidation:

    def test_cross_validate_raises_if_not_fitted(self):
        m = ProphetModel()
        with pytest.raises(RuntimeError, match="fit()"):
            m.cross_validate()

    def test_cross_validate_delegates_to_prophet_cv(self, daily_df):
        m = ProphetModel()
        mock_prophet = MagicMock()
        with patch.object(m, "_build_prophet", return_value=mock_prophet):
            m.fit(daily_df)

        mock_cv_df = pd.DataFrame({
            "ds": pd.date_range("2023-01-01", periods=5),
            "y": [1, 2, 3, 4, 5], "yhat": [1.1, 2.1, 3.1, 4.1, 5.1],
            "yhat_lower": [0.9] * 5, "yhat_upper": [1.3] * 5,
            "cutoff": pd.date_range("2022-12-01", periods=5),
        })

        with patch("core.models.prophet_model.cross_validation", return_value=mock_cv_df) as mock_cv:
            result = m.cross_validate(horizon="30 days", period="15 days")

        mock_cv.assert_called_once()
        assert isinstance(result, pd.DataFrame)

    def test_cross_validation_metrics_returns_rmse_mae_mape(self, daily_df):
        m = ProphetModel()
        mock_prophet = MagicMock()
        with patch.object(m, "_build_prophet", return_value=mock_prophet):
            m.fit(daily_df)

        mock_cv_df = pd.DataFrame({
            "ds":        pd.date_range("2023-01-01", periods=3),
            "y":         [1.0, 2.0, 3.0],
            "yhat":      [1.1, 2.1, 3.1],
            "yhat_lower": [0.9, 1.9, 2.9],
            "yhat_upper": [1.3, 2.3, 3.3],
            "cutoff":    pd.date_range("2022-12-01", periods=3),
        })

        mock_perf = pd.DataFrame({
            "rmse": [2.5], "mae": [1.8], "mape": [0.03],
        })

        with patch("core.models.prophet_model.cross_validation", return_value=mock_cv_df):
            with patch("core.models.prophet_model.performance_metrics", return_value=mock_perf):
                metrics = m.cross_validation_metrics()

        assert "rmse" in metrics
        assert "mae"  in metrics
        assert "mape" in metrics

    def test_cross_validation_metrics_values_are_floats(self, daily_df):
        m = ProphetModel()
        mock_prophet = MagicMock()
        with patch.object(m, "_build_prophet", return_value=mock_prophet):
            m.fit(daily_df)

        mock_cv_df = pd.DataFrame({
            "ds": pd.date_range("2023-01-01", periods=2),
            "y": [1.0, 2.0], "yhat": [1.1, 2.1],
            "yhat_lower": [0.9, 1.9], "yhat_upper": [1.3, 2.3],
            "cutoff": pd.date_range("2022-12-01", periods=2),
        })
        mock_perf = pd.DataFrame({"rmse": [1.0], "mae": [0.8], "mape": [0.02]})

        with patch("core.models.prophet_model.cross_validation", return_value=mock_cv_df):
            with patch("core.models.prophet_model.performance_metrics", return_value=mock_perf):
                metrics = m.cross_validation_metrics()

        assert isinstance(metrics["rmse"], float)
        assert isinstance(metrics["mae"],  float)
        assert isinstance(metrics["mape"], float)

    def test_initial_and_parallel_passed_to_cv(self, daily_df):
        m = ProphetModel()
        mock_prophet = MagicMock()
        with patch.object(m, "_build_prophet", return_value=mock_prophet):
            m.fit(daily_df)

        mock_cv_df = pd.DataFrame({
            "ds": pd.date_range("2023-01-01", periods=2),
            "y": [1.0, 2.0], "yhat": [1.0, 2.0],
            "yhat_lower": [0.9, 1.9], "yhat_upper": [1.1, 2.1],
            "cutoff": pd.date_range("2022-12-01", periods=2),
        })

        with patch("core.models.prophet_model.cross_validation", return_value=mock_cv_df) as mock_cv:
            m.cross_validate(horizon="60 days", period="30 days", initial="90 days", parallel="processes")

        call_kwargs = mock_cv.call_args[1]
        assert call_kwargs["initial"]  == "90 days"
        assert call_kwargs["parallel"] == "processes"



class TestSaveLoad:

    def test_save_raises_if_not_fitted(self, tmp_path):
        m = ProphetModel()
        with pytest.raises(RuntimeError, match="fit()"):
            m.save(str(tmp_path / "model.joblib"))

    def test_save_creates_file(self, daily_df, tmp_path):
        m = ProphetModel()
        mock_prophet = MagicMock()
        with patch.object(m, "_build_prophet", return_value=mock_prophet):
            m.fit(daily_df)

        path = str(tmp_path / "model.joblib")
        with patch("core.models.prophet_model.joblib.dump") as mock_dump:
            m.save(path)
        mock_dump.assert_called_once()

    def test_save_returns_absolute_path(self, daily_df, tmp_path):
        m = ProphetModel()
        mock_prophet = MagicMock()
        with patch.object(m, "_build_prophet", return_value=mock_prophet):
            m.fit(daily_df)

        path = str(tmp_path / "model.joblib")
        with patch("core.models.prophet_model.joblib.dump"):
            returned = m.save(path)

        assert os.path.isabs(returned)

    def test_load_raises_if_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ProphetModel.load(str(tmp_path / "nonexistent.joblib"))

    def test_load_returns_prophet_model_instance(self, daily_df, tmp_path):
        m = ProphetModel()
        mock_prophet = MagicMock()
        with patch.object(m, "_build_prophet", return_value=mock_prophet):
            m.fit(daily_df)

        path = str(tmp_path / "model.joblib")
        with patch("core.models.prophet_model.joblib.dump"):
            with patch("core.models.prophet_model.joblib.load", return_value=m):
                with patch("os.path.exists", return_value=True):
                    loaded = ProphetModel.load(path)

        assert isinstance(loaded, ProphetModel)

    def test_loaded_model_preserves_model_id(self, daily_df, tmp_path):
        m = ProphetModel()
        mock_prophet = MagicMock()
        with patch.object(m, "_build_prophet", return_value=mock_prophet):
            m.fit(daily_df)

        path = str(tmp_path / "model.joblib")
        with patch("core.models.prophet_model.joblib.dump"):
            with patch("core.models.prophet_model.joblib.load", return_value=m):
                with patch("os.path.exists", return_value=True):
                    loaded = ProphetModel.load(path)

        assert loaded.model_id == m.model_id



class TestAssertFitted:

    def test_predict_guard(self):
        m      = ProphetModel()
        future = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=3)})
        with pytest.raises(RuntimeError, match="fit()"):
            m.predict(future)

    def test_make_future_dataframe_guard(self):
        with pytest.raises(RuntimeError, match="fit()"):
            ProphetModel().make_future_dataframe(10)

    def test_cross_validate_guard(self):
        with pytest.raises(RuntimeError, match="fit()"):
            ProphetModel().cross_validate()

    def test_cross_validation_metrics_guard(self):
        with pytest.raises(RuntimeError, match="fit()"):
            ProphetModel().cross_validation_metrics()

    def test_save_guard(self, tmp_path):
        with pytest.raises(RuntimeError, match="fit()"):
            ProphetModel().save(str(tmp_path / "x.joblib"))

    def test_plot_components_guard(self):
        with pytest.raises(RuntimeError, match="fit()"):
            ProphetModel().plot_components(pd.DataFrame())

    def test_error_message_contains_method_name(self):
        m = ProphetModel()
        with pytest.raises(RuntimeError, match="make_future_dataframe"):
            m._assert_fitted("make_future_dataframe")


class TestRepr:

    def test_repr_shows_unfitted_before_fit(self):
        m = ProphetModel()
        assert "unfitted" in repr(m)

    def test_repr_shows_fitted_after_fit(self, daily_df):
        m = ProphetModel()
        with patch.object(m, "_build_prophet", return_value=MagicMock()):
            m.fit(daily_df)
        assert "fitted" in repr(m)

    def test_repr_shows_seasonality_mode(self):
        m = ProphetModel(seasonality_mode="additive")
        assert "additive" in repr(m)

    def test_repr_shows_regressors(self):
        m = ProphetModel(extra_regressors=["promo"])
        assert "promo" in repr(m)

    def test_repr_contains_truncated_model_id(self):
        m = ProphetModel()
        assert m.model_id[:8] in repr(m)



class TestEdgeCases:

    def test_single_row_df_raises(self):
        df = pd.DataFrame({
            "ds": [pd.Timestamp("2023-01-01")],
            "y":  [42.0],
        })
        m = ProphetModel()
        with patch.object(m, "_build_prophet", return_value=MagicMock()):
            with pytest.raises(ValueError, match="Insufficient rows"):
                m.fit(df)

    def test_negative_target_values_do_not_crash_fit(self):
        df = make_daily_df(100)
        df["y"] = df["y"] - 500   # force negatives
        m = ProphetModel()
        mock_prophet = MagicMock()
        with patch.object(m, "_build_prophet", return_value=mock_prophet):
            m.fit(df)  # must not raise
        assert m._is_fitted is True

    def test_all_null_target_raises_value_error(self):
        df = pd.DataFrame({
            "ds": pd.date_range("2023-01-01", periods=10),
            "y":  [np.nan] * 10,
        })
        m = ProphetModel()
        with patch.object(m, "_build_prophet", return_value=MagicMock()):
            with pytest.raises(ValueError):
                m.fit(df)

    def test_fit_is_idempotent(self, daily_df):
        """Calling fit() twice resets the model without error."""
        m = ProphetModel()
        mock1, mock2 = MagicMock(), MagicMock()
        with patch.object(m, "_build_prophet", side_effect=[mock1, mock2]):
            m.fit(daily_df)
            first_id = id(m._model)
            m.fit(daily_df)
            second_id = id(m._model)
        assert first_id != second_id   # model was replaced

    def test_string_datetime_column_is_accepted(self):
        """Dates stored as strings should be parsed to datetime without error."""
        df = pd.DataFrame({
            "ds": ["2023-01-01", "2023-01-02", "2023-01-03",
                   "2023-01-04", "2023-01-05"],
            "y":  [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        m = ProphetModel()
        with patch.object(m, "_build_prophet", return_value=MagicMock()):
            m.fit(df)
        assert m._is_fitted is True

    def test_integer_target_is_accepted(self):
        """Integer y values should be coerced to float without error."""
        df = make_daily_df(50)
        df["y"] = df["y"].astype(int)
        m = ProphetModel()
        with patch.object(m, "_build_prophet", return_value=MagicMock()):
            m.fit(df)
        assert m._is_fitted is True