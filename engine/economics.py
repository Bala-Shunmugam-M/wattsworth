"""Economics: convert energy quantities into monetary terms (INR)."""
from __future__ import annotations

from engine import config


def energy_cost(
    kwh: float,
    tariff_inr_per_kwh: float = config.TARIFF_FLAT_INR_PER_KWH,
) -> float:
    """Return the INR cost of ``kwh`` at a flat tariff.

    Args:
        kwh: Energy in kilowatt-hours.
        tariff_inr_per_kwh: Price per kWh.

    Returns:
        Cost in INR.
    """
    return float(kwh) * float(tariff_inr_per_kwh)


def time_of_use_cost(
    peak_kwh: float,
    offpeak_kwh: float,
    tariff_peak: float = config.TARIFF_PEAK_INR_PER_KWH,
    tariff_offpeak: float = config.TARIFF_OFFPEAK_INR_PER_KWH,
) -> float:
    """Return total INR cost given a peak / off-peak energy split."""
    return float(peak_kwh) * tariff_peak + float(offpeak_kwh) * tariff_offpeak


def simple_payback_months(capex_inr: float, monthly_saving_inr: float) -> float:
    """Return simple payback in months (``inf`` if there is no saving)."""
    if monthly_saving_inr <= 0:
        return float("inf")
    return float(capex_inr) / float(monthly_saving_inr)
