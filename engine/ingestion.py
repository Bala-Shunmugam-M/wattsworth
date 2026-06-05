"""Ingestion & validation for plant-energy data.

Real meter exports are messy: thousands separators, ``"N/A"`` strings, blank cells,
duplicate dates, out-of-range values, gaps. The engine's analytics assume clean,
numeric, de-duplicated, sorted daily data — so this module is the gate that turns a
raw upload into something safe to analyse, and reports exactly what it had to fix.

``validate_and_clean_plant_energy`` never raises on dirty data; it cleans what it
can and returns a :class:`ValidationReport` describing every coercion, drop, clip,
and gap, plus an ``ok`` flag. Pure Python — no Streamlit imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine.data import PLANT_ENERGY_COLUMNS

_ESSENTIAL = ["date", "energy_kwh", "production_tonnes"]
_NUMERIC = ["energy_kwh", "production_tonnes", "ambient_temp_c", "operating_hours", "grid_pf"]
_OPTIONAL = ["ambient_temp_c", "operating_hours", "grid_pf", "tariff_period"]
_NULL_TOKENS = {"", "nan", "na", "n/a", "null", "none", "-"}


@dataclass
class ValidationReport:
    """A record of what ingestion had to do to make a frame analysable.

    Attributes:
        rows_in: Rows received.
        rows_out: Rows surviving cleaning.
        missing_columns: Required columns that were absent.
        dropped_invalid_rows: Rows dropped for missing/invalid essential values.
        duplicate_rows_removed: Duplicate-date rows removed (last kept).
        coerced_cells: Non-numeric cells coerced to NaN during parsing.
        clipped_cells: Out-of-range values clipped to valid bounds.
        date_gaps: Missing days in the daily date range (after cleaning).
        messages: Human-readable notes.
        ok: True if the cleaned frame is usable (has rows and all required columns).
    """

    rows_in: int = 0
    rows_out: int = 0
    missing_columns: list[str] = field(default_factory=list)
    dropped_invalid_rows: int = 0
    duplicate_rows_removed: int = 0
    coerced_cells: int = 0
    clipped_cells: int = 0
    date_gaps: int = 0
    messages: list[str] = field(default_factory=list)
    ok: bool = True


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Coerce a column to numeric, tolerating thousands separators and null tokens.

    Handles object *and* the pandas-3.0 ``str`` dtype (string columns are no longer
    ``object``), stripping commas and mapping null tokens to NaN before parsing.
    """
    if not pd.api.types.is_numeric_dtype(series):
        cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
        cleaned = cleaned.where(~cleaned.str.lower().isin(_NULL_TOKENS), other=np.nan)
        return pd.to_numeric(cleaned, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def validate_and_clean_plant_energy(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate and clean a raw plant-energy frame.

    Coerces numerics, parses dates, normalises ``tariff_period``, clips out-of-range
    values, drops rows with missing essential fields, de-duplicates on date, sorts,
    and detects daily gaps. Never raises on bad data.

    Args:
        df: Raw frame (e.g. from ``pd.read_csv`` of an uploaded file).

    Returns:
        ``(clean_df, report)``. ``clean_df`` has columns
        :data:`~engine.data.PLANT_ENERGY_COLUMNS`; it is empty if the data was
        unusable (see ``report.ok`` and ``report.messages``).
    """
    report = ValidationReport(rows_in=len(df))
    work = df.copy()

    missing = [c for c in _ESSENTIAL if c not in work.columns]
    if missing:
        report.missing_columns = missing
        report.ok = False
        report.messages.append(f"Missing essential column(s): {missing}.")
        return pd.DataFrame(columns=PLANT_ENERGY_COLUMNS), report

    # Add any absent optional columns as empty so the schema is complete.
    for col in _OPTIONAL:
        if col not in work.columns:
            work[col] = np.nan
            report.messages.append(f"Optional column '{col}' was absent; filled with NaN.")

    # Parse dates.
    before_na = work["date"].isna().sum()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    report.coerced_cells += int(work["date"].isna().sum() - before_na)

    # Coerce numerics (counting cells that became NaN).
    for col in _NUMERIC:
        before = work[col].isna().sum()
        work[col] = _coerce_numeric(work[col])
        report.coerced_cells += int(work[col].isna().sum() - before)

    # Normalise tariff_period.
    work["tariff_period"] = (
        work["tariff_period"].astype(str).str.strip().str.lower().where(
            lambda s: s.isin(["peak", "offpeak"]), other=np.nan
        )
    )

    # Range handling: clip what is salvageable, invalidate what is not.
    def _clip(col: str, low: float, high: float) -> None:
        outside = ((work[col] < low) | (work[col] > high)) & work[col].notna()
        report.clipped_cells += int(outside.sum())
        work[col] = work[col].clip(low, high)

    _clip("grid_pf", 0.0, 1.0)
    _clip("operating_hours", 0.0, 24.0)
    _clip("ambient_temp_c", -50.0, 60.0)
    work.loc[work["energy_kwh"] < 0, "energy_kwh"] = np.nan       # negative energy is invalid
    work.loc[work["production_tonnes"] <= 0, "production_tonnes"] = np.nan  # zero can't be a denominator

    # Drop rows missing any essential value.
    before_rows = len(work)
    work = work.dropna(subset=_ESSENTIAL)
    report.dropped_invalid_rows = int(before_rows - len(work))

    # De-duplicate on date (keep the last reading), then sort.
    dup = int(work["date"].duplicated().sum())
    report.duplicate_rows_removed = dup
    work = work.drop_duplicates(subset="date", keep="last").sort_values("date").reset_index(drop=True)

    # Detect daily gaps.
    if len(work) >= 2:
        full = pd.date_range(work["date"].min(), work["date"].max(), freq="D")
        report.date_gaps = int(len(full) - len(work))

    report.rows_out = len(work)
    report.ok = report.rows_out > 0
    if not report.ok:
        report.messages.append("No usable rows remained after cleaning.")

    return work[PLANT_ENERGY_COLUMNS], report
