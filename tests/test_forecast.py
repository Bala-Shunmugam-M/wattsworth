"""Tests for energy forecasting (engine/forecast.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import data, forecast as fc  # noqa: E402


@pytest.fixture(scope="module")
def plant() -> pd.DataFrame:
    return data.generate_synthetic_plant_data(days=540, seed=42)


def test_forecast_shape_and_horizon(plant: pd.DataFrame) -> None:
    result = fc.forecast(plant, horizon_days=21)
    assert len(result.forecast) == 21
    assert len(result.lower) == 21
    assert len(result.upper) == 21
    assert result.method in {"holt-winters", "naive-mean"}


def test_forecast_dates_are_future(plant: pd.DataFrame) -> None:
    result = fc.forecast(plant, horizon_days=14)
    last_obs = pd.to_datetime(plant["date"]).max()
    assert result.forecast.index.min() == last_obs + pd.Timedelta(days=1)
    assert result.forecast.index.is_monotonic_increasing


def test_forecast_band_orders_correctly(plant: pd.DataFrame) -> None:
    result = fc.forecast(plant, horizon_days=14)
    assert (result.lower <= result.forecast + 1e-6).all()
    assert (result.forecast <= result.upper + 1e-6).all()
    assert (result.forecast > 0).all()
    assert (result.lower >= 0).all()


def test_forecast_values_plausible(plant: pd.DataFrame) -> None:
    """Forecast should sit near the recent operating level."""
    result = fc.forecast(plant, horizon_days=14)
    recent_mean = plant["energy_kwh"].tail(60).mean()
    assert 0.5 * recent_mean < result.forecast.mean() < 1.5 * recent_mean


def test_forecast_missing_column_raises(plant: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="must contain"):
        fc.forecast(plant.drop(columns=["energy_kwh"]))


def test_forecast_bad_horizon_raises(plant: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="horizon_days must be"):
        fc.forecast(plant, horizon_days=0)


def test_forecast_too_few_rows_raises() -> None:
    tiny = data.generate_synthetic_plant_data(days=540).head(10)
    with pytest.raises(ValueError, match="at least 14"):
        fc.forecast(tiny)
