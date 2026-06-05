"""Tests for the WattsWorth synthetic data layer (engine/data.py).

Verifies that the generator produces schema-valid data containing the two
ground-truth events the downstream analytics depend on:

* a ~6% energy-intensity step reduction on 2025-09-01 (for IPMVP M&V), and
* five clearly detectable anomaly spike days (for anomaly detection).

The step test excludes the injected spikes and uses a robust median ratio so it
is not skewed by noise or outliers. The anomaly test reconstructs the exact
injected days from ``df.attrs`` and confirms a simple, generic outlier rule
(robust z-score on the residual from a rolling-median baseline) flags every one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the project root importable when pytest is invoked from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import data  # noqa: E402


# --- Fixtures ----------------------------------------------------------------
@pytest.fixture(scope="module")
def plant() -> pd.DataFrame:
    """A reproducible 540-day plant-energy frame."""
    return data.generate_synthetic_plant_data(days=540, seed=42)


@pytest.fixture(scope="module")
def motors() -> pd.DataFrame:
    """A reproducible motor register."""
    return data.generate_motor_register(seed=42)


# --- Schema & shape ----------------------------------------------------------
def test_schema_and_length(plant: pd.DataFrame) -> None:
    """Frame has exactly the required columns and the requested row count."""
    assert list(plant.columns) == data.PLANT_ENERGY_COLUMNS
    assert len(plant) == 540
    assert data.validate_plant_energy(plant)
    assert pd.api.types.is_datetime64_any_dtype(plant["date"])
    assert (plant["energy_kwh"] >= 0).all()


def test_intervention_in_range(plant: pd.DataFrame) -> None:
    """The 2025-09-01 intervention lies inside the generated window."""
    intervention = pd.Timestamp(plant.attrs["intervention_date"])
    assert plant["date"].min() < intervention < plant["date"].max()


# --- Step change (M&V ground truth) -----------------------------------------
def test_six_percent_step_change(plant: pd.DataFrame) -> None:
    """Specific energy consumption drops ~6% after the intervention date.

    Uses median SEC (robust to noise) and excludes the injected anomaly days so
    the spikes cannot distort the before/after comparison.
    """
    intervention = pd.Timestamp(plant.attrs["intervention_date"])
    anomaly_dates = set(plant.attrs["anomaly_dates"])

    df = plant[~plant["date"].isin(anomaly_dates)].copy()
    df["sec"] = df["energy_kwh"] / df["production_tonnes"]

    before = df.loc[df["date"] < intervention, "sec"].median()
    after = df.loc[df["date"] >= intervention, "sec"].median()

    reduction = (before - after) / before
    assert after < before, "Energy intensity should fall after the intervention."
    assert 0.035 < reduction < 0.095, f"Expected ~6% step, got {reduction:.1%}."


# --- Anomalies (anomaly-detection ground truth) ------------------------------
def _driver_expected(df: pd.DataFrame) -> np.ndarray:
    """Return energy expected from the drivers via a quick OLS fit.

    Regresses energy on production, ambient temperature, operating hours, and a
    post-intervention dummy — the same idea the real ``engine.baseline`` module
    uses. This is what a +15-25% spike should be measured *against*, since raw
    energy is dominated by random daily production swings.
    """
    energy = df["energy_kwh"].to_numpy(dtype=float)
    n = len(df)
    intervention = pd.Timestamp(df.attrs.get("intervention_date", "2025-09-01"))
    post = (df["date"] >= intervention).to_numpy().astype(float)
    design = np.column_stack(
        [
            np.ones(n),
            df["production_tonnes"].to_numpy(dtype=float),
            df["ambient_temp_c"].to_numpy(dtype=float),
            df["operating_hours"].to_numpy(dtype=float),
            post,
        ]
    )
    beta, *_ = np.linalg.lstsq(design, energy, rcond=None)
    return design @ beta


def _flag_outliers(df: pd.DataFrame, z_threshold: float = 4.0) -> pd.Series:
    """Flag spikes via robust (MAD-based) z-score of driver-adjusted residuals.

    A single-day spike appears as a large positive residual once the driver
    relationship is removed; the robust z-score keeps the threshold insensitive
    to the spikes' own magnitude.
    """
    energy = df["energy_kwh"].to_numpy(dtype=float)
    resid = energy - _driver_expected(df)
    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    robust_z = (resid - med) / (1.4826 * mad)
    return pd.Series(robust_z > z_threshold, index=df.index)


def test_five_anomalies_present_and_detectable(plant: pd.DataFrame) -> None:
    """All five injected spike days are present and caught by generic outlier logic."""
    anomaly_dates = set(plant.attrs["anomaly_dates"])
    assert len(anomaly_dates) == data.N_ANOMALIES

    flagged_mask = _flag_outliers(plant)
    flagged_dates = set(plant.loc[flagged_mask, "date"])

    # Every injected anomaly must be detected...
    missing = anomaly_dates - flagged_dates
    assert not missing, f"Undetected injected anomalies: {sorted(missing)}"

    # ...without the detector drowning in false positives.
    assert len(flagged_dates) <= data.N_ANOMALIES + 5

    # Injected days really are spikes: >=10% above their driver-predicted energy.
    expected = _driver_expected(plant)
    for d in anomaly_dates:
        i = plant.index[plant["date"] == d][0]
        assert plant.loc[i, "energy_kwh"] >= 1.10 * expected[i]


# --- Motor register ----------------------------------------------------------
def test_motor_register_schema(motors: pd.DataFrame) -> None:
    """Motor register has the required columns and ten assets."""
    assert list(motors.columns) == data.MOTOR_COLUMNS
    assert len(motors) == 10
    assert motors["asset_id"].is_unique


def test_motor_load_profile(motors: pd.DataFrame) -> None:
    """Exactly 3 under-loaded and 2 overloaded motors, per the spec."""
    underloaded = motors[motors["load_factor"] < 0.40]
    overloaded = motors[motors["load_factor"] > 1.0]
    assert len(underloaded) == 3, f"Expected 3 under-loaded, got {len(underloaded)}."
    assert len(overloaded) == 2, f"Expected 2 overloaded, got {len(overloaded)}."


def test_motor_criticality_varied(motors: pd.DataFrame) -> None:
    """Criticality spans all three classes."""
    assert set(motors["criticality"]) == {"A", "B", "C"}


def test_measured_kw_consistent(motors: pd.DataFrame) -> None:
    """measured_kw is consistent with rated_kw * load_factor."""
    expected = (motors["rated_kw"] * motors["load_factor"]).round(1)
    assert np.allclose(motors["measured_kw"], expected, atol=0.15)


# --- Persistence & reproducibility ------------------------------------------
def test_save_synthetic_data_roundtrip(tmp_path: Path) -> None:
    """save_synthetic_data writes valid, reloadable CSVs."""
    plant_path, motors_path = data.save_synthetic_data(data_dir=tmp_path)
    assert plant_path.exists() and motors_path.exists()

    reloaded = data.load_plant_energy(plant_path)
    assert reloaded is not None
    assert data.validate_plant_energy(reloaded)
    assert len(reloaded) == 540


def test_generator_is_reproducible() -> None:
    """Same seed yields identical energy series."""
    a = data.generate_synthetic_plant_data(days=540, seed=7)
    b = data.generate_synthetic_plant_data(days=540, seed=7)
    pd.testing.assert_series_equal(a["energy_kwh"], b["energy_kwh"])
