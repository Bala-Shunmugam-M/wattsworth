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

    The significance test corrects for serial correlation: daily avoided-energy
    values are autocorrelated, so a naive i.i.d. t-test overstates confidence. We
    deflate the sample size to an *effective* count ``n_eff = n·(1-ρ)/(1+ρ)`` using
    the lag-1 autocorrelation ρ, then test against ``t`` with ``n_eff - 1`` df.

    Attributes:
        avoided_kwh: Total avoided energy over the reporting period.
        pct_saving: Saving as a fraction of expected (baseline) energy.
        inr_saved: Monetary saving over the period (INR).
        tonnes_co2_avoided: CO2 avoided over the period (tonnes).
        days: Number of reporting-period days included.
        annual_kwh: Saving projected to a full year (kWh).
        annual_inr: Saving projected to a full year (INR).
        annual_tco2: CO2 avoided projected to a full year (tonnes).
        lag1_autocorr: Lag-1 autocorrelation of daily avoided energy (ρ).
        n_effective: Autocorrelation-adjusted effective sample size.
        t_stat: One-sided t-statistic (using the effective sample size).
        p_value: Autocorrelation-corrected one-sided p-value (mean daily saving > 0).
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
    lag1_autocorr: float = float("nan")
    n_effective: float = float("nan")
    t_stat: float = float("nan")
    p_value: float = float("nan")
    is_significant: bool = False
    avoided_kwh_ci_low: float = float("nan")
    avoided_kwh_ci_high: float = float("nan")
    ci_confidence: float = 0.90


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
    # Correct for serial correlation (daily savings are autocorrelated), which a
    # naive i.i.d. t-test would ignore and so overstate significance.
    daily_values = avoided["avoided_kwh"].to_numpy(dtype=float)
    rho, n_eff, t_stat, p_value = _autocorrelation_corrected_test(daily_values)
    ci_confidence = 0.90
    ci_low, ci_high = _savings_confidence_interval(daily_values, n_eff, days, ci_confidence)

    return SavingsSummary(
        avoided_kwh=total_avoided,
        pct_saving=pct,
        inr_saved=economics.energy_cost(total_avoided, tariff),
        tonnes_co2_avoided=carbon.co2_emissions_tonnes(total_avoided, factor),
        days=days,
        annual_kwh=annual_kwh,
        annual_inr=economics.energy_cost(annual_kwh, tariff),
        annual_tco2=carbon.co2_emissions_tonnes(annual_kwh, factor),
        lag1_autocorr=rho,
        n_effective=n_eff,
        t_stat=t_stat,
        p_value=p_value,
        is_significant=bool(p_value < 0.05) if not np.isnan(p_value) else False,
        avoided_kwh_ci_low=ci_low,
        avoided_kwh_ci_high=ci_high,
        ci_confidence=ci_confidence,
    )


def _savings_confidence_interval(
    daily_values: np.ndarray,
    n_eff: float,
    days: int,
    confidence: float = 0.90,
) -> tuple[float, float]:
    """Return a two-sided confidence interval for *total* avoided energy.

    Uses the autocorrelation-adjusted effective sample size so the band widens with
    serial correlation: margin = t(conf, n_eff-1) * std/sqrt(n_eff) per day, scaled
    to the reporting period. Returns ``(nan, nan)`` when the inputs are degenerate.
    """
    values = np.asarray(daily_values, dtype=float)
    n = values.size
    if n < 3 or np.isnan(n_eff) or n_eff <= 1.0 or np.std(values) == 0 or np.isnan(values).any():
        return float("nan"), float("nan")
    std = float(values.std(ddof=1))
    t_crit = float(stats.t.ppf(1.0 - (1.0 - confidence) / 2.0, df=n_eff - 1.0))
    margin_total = t_crit * (std / np.sqrt(n_eff)) * days
    total = float(values.sum())
    return total - margin_total, total + margin_total


def _autocorrelation_corrected_test(
    daily_values: np.ndarray,
) -> tuple[float, float, float, float]:
    """One-sided t-test that mean daily saving > 0, corrected for autocorrelation.

    Returns ``(lag1_autocorr, n_effective, t_stat, p_value)``. The effective sample
    size ``n_eff = n·(1-ρ)/(1+ρ)`` deflates the i.i.d. count by the lag-1
    autocorrelation ρ, and the t-test uses ``n_eff - 1`` degrees of freedom.
    """
    values = np.asarray(daily_values, dtype=float)
    n = values.size
    if n < 3 or np.std(values) == 0 or np.isnan(values).any():
        return float("nan"), float("nan"), float("nan"), float("nan")

    mean = float(values.mean())
    std = float(values.std(ddof=1))
    centred = values - mean
    denom = float(np.sum(centred**2))
    rho = float(np.sum(centred[:-1] * centred[1:]) / denom) if denom > 0 else 0.0
    rho = float(min(max(rho, -0.99), 0.99))

    n_eff = n * (1.0 - rho) / (1.0 + rho)
    n_eff = float(min(max(n_eff, 2.0), float(n)))

    t_stat = mean / (std / np.sqrt(n_eff))
    p_value = float(stats.t.sf(t_stat, df=n_eff - 1.0))  # one-sided: mean > 0
    return rho, n_eff, float(t_stat), p_value
