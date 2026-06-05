"""Tests for the ISO 50001 baseline engine (engine/baseline.py).

A correct energy baseline is fit on a *clean, single-regime* window — the
pre-intervention period with known anomaly days excluded. These tests follow
that practice: the ``model`` fixture is fit on the clean window, and the M&V
test then shows that this baseline predicts ~6% more energy than the plant
actually used in the post-intervention reporting period.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the project root importable when pytest is invoked from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import baseline, data  # noqa: E402

_BASELINE_END = "2025-08-31"  # last day before the 2025-09-01 intervention


# --- Fixtures ----------------------------------------------------------------
@pytest.fixture(scope="module")
def plant() -> pd.DataFrame:
    """A reproducible 540-day plant-energy frame (both regimes)."""
    return data.generate_synthetic_plant_data(days=540, seed=42)


@pytest.fixture(scope="module")
def clean(plant: pd.DataFrame) -> pd.DataFrame:
    """The clean baseline window: pre-intervention, anomaly days removed."""
    anomalies = set(plant.attrs["anomaly_dates"])
    return plant[
        (plant["date"] <= pd.Timestamp(_BASELINE_END)) & (~plant["date"].isin(anomalies))
    ].copy()


@pytest.fixture(scope="module")
def model(clean: pd.DataFrame) -> baseline.BaselineModel:
    """A baseline fitted on the clean window."""
    return baseline.fit_baseline(clean)


# --- Fit quality -------------------------------------------------------------
def test_fit_quality(model: baseline.BaselineModel, clean: pd.DataFrame) -> None:
    """The baseline explains the data well and is IPMVP-acceptable."""
    assert model.r_squared > 0.85, f"R2 too low: {model.r_squared:.3f}"
    assert model.cv_rmse < 0.20, f"CV(RMSE) too high: {model.cv_rmse:.3f}"
    assert model.n_obs == len(clean)
    assert "const" in model.coefficients


def test_all_real_drivers_retained(model: baseline.BaselineModel) -> None:
    """All three genuine drivers survive significance filtering on this data."""
    assert set(model.drivers) == set(data.config.DEFAULT_DRIVERS)
    assert not model.dropped


def test_coefficients_have_expected_sign(model: baseline.BaselineModel) -> None:
    """Energy rises with production, temperature, and runtime (positive slopes)."""
    for driver in ("production_tonnes", "ambient_temp_c", "operating_hours"):
        assert model.coefficients[driver] > 0, f"{driver} should have a positive coef."


# --- The core M&V job: detect the 6% step ------------------------------------
def test_baseline_detects_six_percent_step(plant: pd.DataFrame) -> None:
    """Trained on the pre-intervention period, the model over-predicts post-energy by ~6%.

    This is IPMVP Option C in miniature: the baseline encodes the *old* energy
    behaviour, so on the reporting (post) period actual energy should fall ~6%
    short of predicted. Anomaly days are excluded so spikes don't distort totals.
    """
    intervention = pd.Timestamp(plant.attrs["intervention_date"])
    anomaly_dates = set(plant.attrs["anomaly_dates"])
    start = plant["date"].min().strftime("%Y-%m-%d")

    pre_model = baseline.fit_baseline(plant, baseline_period=(start, _BASELINE_END))
    assert pre_model.n_obs < len(plant), "Baseline period should restrict rows."

    post = plant[(plant["date"] >= intervention) & (~plant["date"].isin(anomaly_dates))]
    expected = baseline.predict_expected(pre_model, post)
    actual = post["energy_kwh"]

    saving_fraction = 1.0 - actual.sum() / expected.sum()
    assert 0.03 < saving_fraction < 0.09, f"Expected ~6% saving, got {saving_fraction:.1%}."


# --- Backward elimination ----------------------------------------------------
def test_insignificant_driver_dropped(clean: pd.DataFrame) -> None:
    """A pure-noise driver is removed; the three real drivers remain."""
    df = clean.copy()
    rng = np.random.default_rng(123)
    df["random_noise"] = rng.normal(0.0, 1.0, len(df))

    candidate = [*data.config.DEFAULT_DRIVERS, "random_noise"]
    fitted = baseline.fit_baseline(df, drivers=candidate)

    assert "random_noise" in fitted.dropped
    assert "random_noise" not in fitted.drivers
    assert set(fitted.drivers) == set(data.config.DEFAULT_DRIVERS)


# --- Prediction sanity -------------------------------------------------------
def test_predictions_reasonable(clean: pd.DataFrame, model: baseline.BaselineModel) -> None:
    """Predictions are positive, correctly scaled, and track actuals tightly."""
    expected = baseline.predict_expected(model, clean)
    actual = clean["energy_kwh"]

    assert expected.name == "expected_energy_kwh"
    assert len(expected) == len(clean)
    assert (expected > 0).all()
    # Mean prediction within 2% of mean actual (in-sample, so should be very close).
    assert abs(expected.mean() - actual.mean()) / actual.mean() < 0.02
    # Strong correlation with actual energy.
    assert np.corrcoef(expected, actual)[0, 1] > 0.85


def test_predict_on_new_rows(model: baseline.BaselineModel) -> None:
    """The model predicts sensibly on a hand-built row."""
    row = pd.DataFrame(
        {"production_tonnes": [92.0], "ambient_temp_c": [32.0], "operating_hours": [22.5]}
    )
    pred = baseline.predict_expected(model, row)
    assert 5_000 < float(pred.iloc[0]) < 15_000  # plausible daily kWh for this plant


# --- Error handling ----------------------------------------------------------
def test_missing_target_or_driver_raises(plant: pd.DataFrame) -> None:
    """Missing required columns raise a clear ValueError."""
    with pytest.raises(ValueError, match="Missing required column"):
        baseline.fit_baseline(plant.drop(columns=["production_tonnes"]))


def test_empty_baseline_period_raises(plant: pd.DataFrame) -> None:
    """A baseline period selecting no rows raises ValueError."""
    with pytest.raises(ValueError, match="selected 0 rows"):
        baseline.fit_baseline(plant, baseline_period=("2030-01-01", "2030-12-31"))


def test_predict_requires_fitted_model() -> None:
    """Predicting with an unfitted model raises ValueError."""
    with pytest.raises(ValueError, match="not fitted"):
        baseline.predict_expected(baseline.BaselineModel(), pd.DataFrame())


def test_no_drivers_raises(plant: pd.DataFrame) -> None:
    """An empty driver list raises ValueError."""
    with pytest.raises(ValueError, match="At least one"):
        baseline.fit_baseline(plant, drivers=[])
