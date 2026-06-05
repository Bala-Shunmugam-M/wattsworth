"""Tests for the ingestion / validation layer (engine/ingestion.py).

These are the messy-data tests the audit flagged as missing: thousands separators,
null tokens, duplicate dates, out-of-range values, missing columns, and gaps —
the inputs that crash the raw loaders. The cleaner must tolerate all of them and
report what it did.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine import data, ingestion


def test_clean_data_passes_through_unchanged() -> None:
    """Already-clean synthetic data survives with no drops or coercions."""
    df = data.generate_synthetic_plant_data(120, 42)
    clean, report = ingestion.validate_and_clean_plant_energy(df)
    assert report.ok
    assert report.rows_out == len(df)
    assert report.dropped_invalid_rows == 0
    assert report.coerced_cells == 0
    assert list(clean.columns) == data.PLANT_ENERGY_COLUMNS


def test_thousands_separators_and_null_tokens() -> None:
    """'1,200' parses to 1200; 'N/A'/'' become NaN and drop if essential."""
    raw = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "energy_kwh": ["1,200", "N/A", "1300"],
            "production_tonnes": ["90", "92", ""],
            "ambient_temp_c": [30, 31, 32],
            "operating_hours": [22, 22, 23],
            "grid_pf": [0.85, 0.86, 0.84],
            "tariff_period": ["Peak", "OFFPEAK", "peak"],
        }
    )
    clean, report = ingestion.validate_and_clean_plant_energy(raw)
    assert clean.loc[clean["date"] == pd.Timestamp("2025-01-01"), "energy_kwh"].iloc[0] == 1200.0
    assert report.coerced_cells >= 2          # 'N/A' energy and '' production
    assert report.dropped_invalid_rows == 2   # rows 2 and 3 lose an essential value
    assert set(clean["tariff_period"]) <= {"peak", "offpeak"}


def test_duplicate_dates_deduplicated() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-01", "2025-01-02"],
            "energy_kwh": [1000, 1100, 1200],
            "production_tonnes": [90, 91, 92],
            "ambient_temp_c": [30, 30, 31],
            "operating_hours": [22, 22, 23],
            "grid_pf": [0.85, 0.85, 0.86],
            "tariff_period": ["peak", "peak", "offpeak"],
        }
    )
    clean, report = ingestion.validate_and_clean_plant_energy(raw)
    assert report.duplicate_rows_removed == 1
    # Last reading kept for the duplicated date.
    assert clean.loc[clean["date"] == pd.Timestamp("2025-01-01"), "energy_kwh"].iloc[0] == 1100.0


def test_out_of_range_values_clipped_or_invalidated() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02"],
            "energy_kwh": [-50, 1200],          # negative -> invalid -> row dropped
            "production_tonnes": [90, 92],
            "ambient_temp_c": [30, 31],
            "operating_hours": [30, 22],        # 30h -> clipped to 24
            "grid_pf": [1.4, 0.86],             # 1.4 -> clipped to 1.0
            "tariff_period": ["peak", "offpeak"],
        }
    )
    clean, report = ingestion.validate_and_clean_plant_energy(raw)
    assert report.clipped_cells >= 2           # hours and pf on row 1
    assert report.dropped_invalid_rows == 1    # negative-energy row dropped
    assert (clean["operating_hours"] <= 24).all()
    assert (clean["grid_pf"] <= 1.0).all()


def test_missing_essential_column_is_fatal() -> None:
    raw = pd.DataFrame({"date": ["2025-01-01"], "energy_kwh": [1000]})  # no production
    clean, report = ingestion.validate_and_clean_plant_energy(raw)
    assert not report.ok
    assert "production_tonnes" in report.missing_columns
    assert clean.empty


def test_missing_optional_column_filled() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02"],
            "energy_kwh": [1000, 1100],
            "production_tonnes": [90, 91],
        }
    )
    clean, report = ingestion.validate_and_clean_plant_energy(raw)
    assert report.ok
    assert "grid_pf" in clean.columns
    assert clean["grid_pf"].isna().all()


def test_gaps_detected_and_sorted() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2025-01-05", "2025-01-01", "2025-01-02"],  # unsorted + gap (3rd,4th missing)
            "energy_kwh": [1200, 1000, 1100],
            "production_tonnes": [92, 90, 91],
            "ambient_temp_c": [31, 30, 30],
            "operating_hours": [23, 22, 22],
            "grid_pf": [0.86, 0.85, 0.85],
            "tariff_period": ["peak", "peak", "offpeak"],
        }
    )
    clean, report = ingestion.validate_and_clean_plant_energy(raw)
    assert clean["date"].is_monotonic_increasing
    assert report.date_gaps == 2               # Jan 3 and Jan 4 missing


def test_empty_frame_is_not_ok() -> None:
    clean, report = ingestion.validate_and_clean_plant_energy(
        pd.DataFrame(columns=data.PLANT_ENERGY_COLUMNS)
    )
    assert not report.ok
    assert report.rows_out == 0


def test_realistic_data_cleans_with_gaps() -> None:
    """The realistic dataset (which has missing days) reports gaps but stays usable."""
    df = data.generate_realistic_plant_data(540, 42)
    clean, report = ingestion.validate_and_clean_plant_energy(df)
    assert report.ok
    assert report.date_gaps > 0                # it has injected missing days
    assert report.rows_out == len(df)
