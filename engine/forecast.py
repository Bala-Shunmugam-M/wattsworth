"""Energy / load forecasting.

Forecasts daily energy with Holt-Winters exponential smoothing (additive trend,
optional weekly seasonality) — a robust univariate method that needs no future
driver inputs, ideal for a short-horizon operational outlook. A constant
confidence band is derived from in-sample residual scatter.

For a *driver-based* forecast (energy given a planned production / weather
schedule), use :func:`engine.baseline.predict_expected` with the future drivers.

Pure Python — no Streamlit imports.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from engine import config

_SEASONAL_PERIOD = 7  # weekly
_Z95 = 1.96


@dataclass
class ForecastResult:
    """Forecast output.

    Attributes:
        forecast: Predicted energy indexed by future date.
        lower: Lower 95% confidence bound.
        upper: Upper 95% confidence bound.
        method: Which model produced the forecast ('holt-winters' or 'naive-mean').
    """

    forecast: pd.Series
    lower: pd.Series
    upper: pd.Series
    method: str = "holt-winters"


def forecast(
    df: pd.DataFrame,
    horizon_days: int = 14,
    target: str = config.TARGET_COLUMN,
    date_col: str = "date",
) -> ForecastResult:
    """Forecast ``target`` for ``horizon_days`` beyond the last observation.

    Args:
        df: Plant-energy frame with ``date_col`` and ``target``.
        horizon_days: Number of future days to predict.
        target: Energy column to forecast.
        date_col: Date column name.

    Returns:
        A :class:`ForecastResult` whose series are indexed by future dates.

    Raises:
        ValueError: If columns are missing, ``horizon_days`` < 1, or there are
            too few observations (< 14) to fit.
    """
    if target not in df.columns or date_col not in df.columns:
        raise ValueError(f"Frame must contain '{date_col}' and '{target}'.")
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1.")

    work = df[[date_col, target]].copy()
    work[date_col] = pd.to_datetime(work[date_col])
    series = (
        work.sort_values(date_col)
        .set_index(date_col)[target]
        .astype(float)
        .asfreq("D")
        .interpolate()
    )
    if len(series) < 14:
        raise ValueError(f"Need at least 14 observations to forecast (got {len(series)}).")

    last_date = series.index[-1]
    future_index = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    try:
        use_seasonal = len(series) >= 2 * _SEASONAL_PERIOD
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted_model = ExponentialSmoothing(
                series,
                trend="add",
                damped_trend=True,
                seasonal="add" if use_seasonal else None,
                seasonal_periods=_SEASONAL_PERIOD if use_seasonal else None,
                initialization_method="estimated",
            ).fit()
        point = fitted_model.forecast(horizon_days)
        resid_std = float(np.std(series.to_numpy() - fitted_model.fittedvalues.to_numpy()))
        method = "holt-winters"
    except Exception:  # noqa: BLE001 — any fit failure falls back to a safe estimate
        recent = series.iloc[-30:]
        point = pd.Series(float(recent.mean()), index=future_index)
        resid_std = float(np.std(recent.to_numpy()))
        method = "naive-mean"

    point = pd.Series(np.asarray(point, dtype=float), index=future_index, name="forecast_kwh")
    margin = _Z95 * resid_std
    lower = (point - margin).clip(lower=0.0).rename("lower_kwh")
    upper = (point + margin).rename("upper_kwh")
    return ForecastResult(forecast=point, lower=lower, upper=upper, method=method)
