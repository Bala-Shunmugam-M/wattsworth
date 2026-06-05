"""IPMVP Option C measurement & verification (savings + CUSUM).

Quantifies avoided energy by comparing actual consumption in a reporting period
against the baseline model's expectation under the same operating conditions.
This is IPMVP Option C ("whole-facility"): the baseline encodes *how the plant
used to behave*, so

    avoided energy = expected (baseline) - actual (reporting)

A positive number means the plant used less than the old baseline predicted —
a real saving once you confirm it exceeds the baseline's own noise. That last
check (a one-sided t-test on daily avoided energy) is what makes the claim
defensible rather than wishful.

Pure Python — no Streamlit imports.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats

from engine import carbon, config, economics
from engine.baseline import predict_expected

if TYPE_CHECKING:
    from engine.baseline import BaselineModel


@dataclass
class SavingsSummary:
    """Headline M&V results for a reporting period.

    Attributes:
        avoided_kwh: Total avoided energy over the reporting period.
        pct_saving: Saving as a fraction of expected (baseline) energy.
        inr_saved: Monetary saving over the period (INR).
        tonnes_co2_avoided: CO2 avoided over the period (tonnes).
        days: Number of reporting-period days included.
        annual_kwh: Saving projected to a full year (kWh).
        annual_inr: Saving projected to a full year (INR).
        annual_tco2: CO2 avoided projected to a full year (tonnes).
        t_stat: One-sided t-statistic testing mean daily avoided energy > 0.
        p_value: p-value of that test (smaller = more confident the saving is real).
        is_significant: True if ``p_value < 0.05``.
    """

    avoided_kwh: float = 0.0
    pct_saving: float = 0.0
    inr_saved: float = 0.0
    tonnes_co2_avoided: float = 0.0
    days: int = 0
    annual_kwh: float = 0.0
    annual_inr: float = 0.0
    annual_tco2: float = 0.0
    t_stat: float = float("nan")
    p_value: float = float("nan")
    is_significant: bool = False


def compute_savings(
    df: pd.DataFrame,
    model: "BaselineModel",
    intervention_date: str,
    target: str = config.TARGET_COLUMN,
    exclude_dates: Optional[Iterable] = None,
) -> pd.DataFrame:
    """Return per-day avoided energy for the reporting period.

    The reporting period is every row on or after ``intervention_date``. For each
    day it computes the baseline-expected energy (from ``model`` and that day's
    drivers) and subtracts the actual, giving the avoided energy.

    Args:
        df: Plant-energy frame with ``date``, ``target``, and the model's drivers.
        model: A fitted :class:`~engine.baseline.BaselineModel`.
        intervention_date: ISO date; the reporting period starts here (inclusive).
        target: Name of the actual-energy column.
        exclude_dates: Optional dates to drop (e.g. known anomaly days) so spikes
            don't distort the saving.

    Returns:
        DataFrame with columns ``date``, ``actual_kwh``, ``expected_kwh``,
        ``avoided_kwh``, ``cumulative_avoided_kwh`` (one row per reporting day).

    Raises:
        ValueError: If the model is unfitted, ``date`` is missing, or the
            reporting period is empty.
    """
    if model.model is None:
        raise ValueError("BaselineModel is not fitted (model is None).")
    if "date" not in df.columns:
        raise ValueError("Frame has no 'date' column.")

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    cutoff = pd.Timestamp(intervention_date)
    post = work.loc[work["date"] >= cutoff]

    if exclude_dates is not None:
        excluded = {pd.Timestamp(d) for d in exclude_dates}
        post = post.loc[~post["date"].isin(excluded)]

    if post.empty:
        raise ValueError(
            f"No reporting-period rows on/after {cutoff.date()} "
            f"(data ends {work['date'].max().date()})."
        )

    expected = predict_expected(model, post)
    actual = post[target].astype(float)
    avoided = expected.to_numpy() - actual.to_numpy()

    out = pd.DataFrame(
        {
            "date": post["date"].to_numpy(),
            "actual_kwh": actual.to_numpy(),
            "expected_kwh": expected.to_numpy(),
            "avoided_kwh": avoided,
        }
    )
    out["cumulative_avoided_kwh"] = out["avoided_kwh"].cumsum()
    return out.reset_index(drop=True)


def cusum(actual: pd.Series, expected: pd.Series) -> pd.Series:
    """Return the cumulative sum of ``(actual - expected)``.

    The classic energy-management CUSUM. A sustained downward slope marks
    persistent savings; a sharp knee marks the moment behaviour changed.
    """
    return (actual - expected).cumsum()


def summarize_savings(
    avoided: pd.DataFrame,
    tariff_inr_per_kwh: float | None = None,
    emission_factor: float | None = None,
) -> SavingsSummary:
    """Aggregate per-day savings into a :class:`SavingsSummary`.

    Computes totals, a full-year projection, and a one-sided t-test of whether
    mean daily avoided energy is significantly greater than zero.

    Args:
        avoided: Output of :func:`compute_savings`.
        tariff_inr_per_kwh: Energy price; defaults to the flat configured tariff.
        emission_factor: kg CO2 per kWh; defaults to the configured grid factor.

    Returns:
        A populated :class:`SavingsSummary`.

    Raises:
        ValueError: If ``avoided`` is empty.
    """
    if avoided.empty:
        raise ValueError("Cannot summarize: 'avoided' frame is empty.")

    tariff = tariff_inr_per_kwh if tariff_inr_per_kwh is not None else config.TARIFF_FLAT_INR_PER_KWH
    factor = emission_factor if emission_factor is not None else config.GRID_EMISSION_FACTOR_KG_PER_KWH

    total_avoided = float(avoided["avoided_kwh"].sum())
    expected_total = float(avoided["expected_kwh"].sum())
    pct = total_avoided / expected_total if expected_total else 0.0

    days = int(len(avoided))
    daily = total_avoided / days if days else 0.0
    annual_kwh = daily * 365.0

    # Is the daily saving significantly greater than zero, or just noise?
    daily_values = avoided["avoided_kwh"].to_numpy(dtype=float)
    if days >= 2 and np.std(daily_values) > 0:
        t_stat, p_value = stats.ttest_1samp(daily_values, 0.0, alternative="greater")
        t_stat, p_value = float(t_stat), float(p_value)
    else:
        t_stat, p_value = float("nan"), float("nan")

    return SavingsSummary(
        avoided_kwh=total_avoided,
        pct_saving=pct,
        inr_saved=economics.energy_cost(total_avoided, tariff),
        tonnes_co2_avoided=carbon.co2_emissions_tonnes(total_avoided, factor),
        days=days,
        annual_kwh=annual_kwh,
        annual_inr=economics.energy_cost(annual_kwh, tariff),
        annual_tco2=carbon.co2_emissions_tonnes(annual_kwh, factor),
        t_stat=t_stat,
        p_value=p_value,
        is_significant=bool(p_value < 0.05) if not np.isnan(p_value) else False,
    )
