"""Tests for the hard-mode (realistic) dataset and the engine's honest degradation.

The clean generator produces data that is exactly linear in the drivers, so every
method looks perfect — circular validation. These tests use
``generate_realistic_plant_data`` (non-linear, autocorrelated, collinear, with
missing days) to prove the engine still works on harder data *and* that its
diagnostics correctly flag the degradation (lower R², Durbin-Watson below 2),
rather than silently pretending everything is fine.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import baseline, data, mv  # noqa: E402

_BASELINE_END = "2025-08-31"
_INTERVENTION = "2025-09-01"


@pytest.fixture(scope="module")
def realistic() -> pd.DataFrame:
    return data.generate_realistic_plant_data(days=540, seed=42)


@pytest.fixture(scope="module")
def pre_model(realistic: pd.DataFrame) -> baseline.BaselineModel:
    pre = realistic[realistic["date"] <= pd.Timestamp(_BASELINE_END)]
    return baseline.fit_baseline(pre)


# --- The data really is harder -----------------------------------------------
def test_has_missing_days(realistic: pd.DataFrame) -> None:
    """~2.5% of days are dropped as missing (sensor outages)."""
    assert 510 < len(realistic) < 535
    assert list(realistic.columns) == data.PLANT_ENERGY_COLUMNS


def test_drivers_are_collinear(realistic: pd.DataFrame) -> None:
    """Production and operating hours share a latent utilisation factor."""
    corr = realistic["production_tonnes"].corr(realistic["operating_hours"])
    assert corr > 0.45  # real collinearity, unlike the independent clean drivers


def test_anomalies_and_intervention_present(realistic: pd.DataFrame) -> None:
    assert len(realistic.attrs["anomaly_dates"]) == data.N_ANOMALIES
    assert realistic.attrs["mode"] == "realistic"


# --- The engine degrades HONESTLY (not silently) -----------------------------
def test_baseline_r2_is_lower_than_clean(pre_model: baseline.BaselineModel) -> None:
    """R² is modest on realistic data — and clearly below the clean-data ~0.92."""
    clean_pre = data.generate_synthetic_plant_data(540, 42)
    clean_pre = clean_pre[clean_pre["date"] <= pd.Timestamp(_BASELINE_END)]
    clean_model = baseline.fit_baseline(clean_pre)
    assert pre_model.r_squared < clean_model.r_squared
    assert 0.40 < pre_model.r_squared < 0.85


def test_cv_rmse_still_passes_ipmvp_gate(pre_model: baseline.BaselineModel) -> None:
    """Despite a modest R², CV(RMSE) still clears the IPMVP/ASHRAE 20% gate."""
    assert pre_model.cv_rmse < 0.15
    assert pre_model.cv_rmse_oos is not None
    assert pre_model.cv_rmse_oos < 0.20


def test_durbin_watson_flags_autocorrelation(pre_model: baseline.BaselineModel) -> None:
    """The AR(1) noise shows up as a Durbin-Watson well below 2."""
    assert pre_model.durbin_watson < 1.6


# --- M&V is still robust on realistic data -----------------------------------
def test_mv_still_recovers_step(realistic: pd.DataFrame, pre_model: baseline.BaselineModel) -> None:
    """The 6% saving is recovered even with non-linearity, gaps, and autocorrelation."""
    avoided = mv.compute_savings(
        realistic, pre_model, _INTERVENTION, exclude_dates=realistic.attrs["anomaly_dates"]
    )
    summary = mv.summarize_savings(avoided)
    assert 0.03 < summary.pct_saving < 0.09
    # Autocorrelation is real here, so the effective sample size shrinks.
    assert summary.lag1_autocorr > 0.2
    assert summary.n_effective < 0.8 * summary.days
    # The 6% effect is still large enough to be significant after the correction.
    assert summary.is_significant
