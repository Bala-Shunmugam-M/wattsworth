"""Carbon accounting: convert energy to CO2 and compute abatement cost."""
from __future__ import annotations

from engine import config


def co2_emissions_kg(
    kwh: float,
    emission_factor_kg_per_kwh: float = config.GRID_EMISSION_FACTOR_KG_PER_KWH,
) -> float:
    """Return CO2 emissions (kg) for ``kwh`` of grid electricity."""
    return float(kwh) * float(emission_factor_kg_per_kwh)


def co2_emissions_tonnes(
    kwh: float,
    emission_factor_kg_per_kwh: float = config.GRID_EMISSION_FACTOR_KG_PER_KWH,
) -> float:
    """Return CO2 emissions (tonnes) for ``kwh`` of grid electricity."""
    return co2_emissions_kg(kwh, emission_factor_kg_per_kwh) / 1000.0


def abatement_cost_inr_per_tonne(capex_inr: float, tonnes_co2_avoided: float) -> float:
    """Return INR cost per tonne of CO2 avoided (``inf`` if none avoided)."""
    if tonnes_co2_avoided <= 0:
        return float("inf")
    return float(capex_inr) / float(tonnes_co2_avoided)
