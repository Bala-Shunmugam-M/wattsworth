"""Tests for IPMVP Option C measurement & verification (engine/mv.py).

Confirms the engine recovers the injected ~6% saving, expresses it in energy /
rupees / CO2, projects it to a year, and — crucially — judges it statistically
significant rather than noise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import baseline, carbon, config, data, economics, mv  # noqa: E402

_BASELINE_END = "2025-08-31"
_INTERVENTION = "2025-09-01"


# --- Fixtures ----------------------------------------------------------------
@pytest.fixture(scope="module")
def plant() -> pd.DataFrame:
    return data.generate_synthetic_plant_data(days=540, seed=42)


@pytest.fixture(scope="module")
def anomalies(plant: pd.DataFrame) -> list[pd.Timestamp]:
    return list(plant.attrs["anomaly_dates"])


@pytest.fixture(scope="module")
def model(plant: pd.DataFrame, anomalies: list[pd.Timestamp]) -> baseline.BaselineModel:
    """Baseline fit on the clean pre-intervention window."""
    clean = plant[
        (plant["date"] <= pd.Timestamp(_BASELINE_END)) & (~plant["date"].isin(set(anomalies)))
    ]
    return baseline.fit_baseline(clean)


@pytest.fixture(scope="module")
def avoided(plant: pd.DataFrame, model: baseline.BaselineModel, anomalies) -> pd.DataFrame:
    """Per-day savings for the reporting period, anomalies excluded."""
    return mv.compute_savings(plant, model, _INTERVENTION, exclude_dates=anomalies)


@pytest.fixture(scope="module")
def summary(avoided: pd.DataFrame) -> mv.SavingsSummary:
    return mv.summarize_savings(avoided)


# --- compute_savings ---------------------------------------------------------
def test_compute_savings_shape(avoided: pd.DataFrame) -> None:
    """Output has the expected columns and one row per reporting day."""
    expected_cols = {"date", "actual_kwh", "expected_kwh", "avoided_kwh", "cumulative_avoided_kwh"}
    assert expected_cols.issubset(avoided.columns)
    assert len(avoided) > 250  # ~9 months of post-intervention days


def test_reporting_period_starts_at_intervention(avoided: pd.DataFrame) -> None:
    """No reporting row predates the intervention date."""
    assert avoided["date"].min() >= pd.Timestamp(_INTERVENTION)


def test_cumulative_is_running_total(avoided: pd.DataFrame) -> None:
    """The cumulative column is the running sum of daily avoided energy."""
    assert avoided["cumulative_avoided_kwh"].iloc[-1] == pytest.approx(
        avoided["avoided_kwh"].sum()
    )


# --- summarize_savings: the 6% recovery --------------------------------------
def test_recovers_six_percent_saving(summary: mv.SavingsSummary) -> None:
    """The engine recovers the injected ~6% energy-intensity reduction."""
    assert 0.04 < summary.pct_saving < 0.08, f"Got {summary.pct_saving:.1%}"
    assert summary.avoided_kwh > 0


def test_savings_are_statistically_significant(summary: mv.SavingsSummary) -> None:
    """The saving is real, not noise: one-sided t-test is highly significant."""
    assert summary.is_significant
    assert summary.p_value < 0.001
    assert summary.t_stat > 0


def test_money_and_carbon_consistent(summary: mv.SavingsSummary) -> None:
    """INR and CO2 follow from avoided energy via the configured factors."""
    assert summary.inr_saved == pytest.approx(
        economics.energy_cost(summary.avoided_kwh, config.TARIFF_FLAT_INR_PER_KWH)
    )
    assert summary.tonnes_co2_avoided == pytest.approx(
        carbon.co2_emissions_tonnes(summary.avoided_kwh, config.GRID_EMISSION_FACTOR_KG_PER_KWH)
    )
    assert summary.inr_saved > 0
    assert summary.tonnes_co2_avoided > 0


def test_annual_projection(summary: mv.SavingsSummary) -> None:
    """Annual figures scale the period saving to 365 days."""
    daily = summary.avoided_kwh / summary.days
    assert summary.annual_kwh == pytest.approx(daily * 365.0)
    assert summary.annual_inr > summary.inr_saved  # 365 days > ~9-month period
    assert summary.annual_tco2 > summary.tonnes_co2_avoided


def test_custom_tariff_and_factor() -> None:
    """summarize_savings honours custom tariff / emission factor."""
    avoided = pd.DataFrame(
        {
            "date": pd.date_range("2025-09-01", periods=10, freq="D"),
            "actual_kwh": [900.0] * 10,
            "expected_kwh": [1000.0] * 10,
            "avoided_kwh": [100.0] * 10,
            "cumulative_avoided_kwh": np.cumsum([100.0] * 10),
        }
    )
    s = mv.summarize_savings(avoided, tariff_inr_per_kwh=10.0, emission_factor=0.8)
    assert s.avoided_kwh == pytest.approx(1000.0)
    assert s.inr_saved == pytest.approx(10_000.0)        # 1000 kWh * 10
    assert s.tonnes_co2_avoided == pytest.approx(0.8)    # 1000 * 0.8 / 1000


# --- CUSUM -------------------------------------------------------------------
def test_cusum_trends_down_under_savings(plant: pd.DataFrame, model: baseline.BaselineModel) -> None:
    """CUSUM of (actual - expected) ends negative when the plant is saving."""
    post = plant[plant["date"] >= pd.Timestamp(_INTERVENTION)]
    expected = baseline.predict_expected(model, post)
    series = mv.cusum(post["energy_kwh"], expected)
    assert series.iloc[-1] < 0  # actual below expected => savings


# --- exclude_dates & errors --------------------------------------------------
def test_exclude_dates_removes_anomalies(
    plant: pd.DataFrame, model: baseline.BaselineModel, anomalies
) -> None:
    """Excluded anomaly days are absent from the reporting frame."""
    avoided = mv.compute_savings(plant, model, _INTERVENTION, exclude_dates=anomalies)
    post_anomalies = [d for d in anomalies if d >= pd.Timestamp(_INTERVENTION)]
    assert not set(avoided["date"]).intersection(post_anomalies)


def test_empty_reporting_period_raises(plant: pd.DataFrame, model: baseline.BaselineModel) -> None:
    """A future intervention date leaves no reporting rows."""
    with pytest.raises(ValueError, match="No reporting-period rows"):
        mv.compute_savings(plant, model, "2030-01-01")


def test_unfitted_model_raises(plant: pd.DataFrame) -> None:
    """compute_savings rejects an unfitted model."""
    with pytest.raises(ValueError, match="not fitted"):
        mv.compute_savings(plant, baseline.BaselineModel(), _INTERVENTION)


def test_summarize_empty_raises() -> None:
    """summarize_savings rejects an empty frame."""
    with pytest.raises(ValueError, match="empty"):
        mv.summarize_savings(pd.DataFrame(columns=["avoided_kwh", "expected_kwh"]))
