"""Number/string formatting for reports (pure functions)."""
from __future__ import annotations

import math
from typing import Mapping

from engine import config


def inr_lakhs(value: float, ascii_only: bool = False) -> str:
    """Format INR in lakhs, e.g. 1_500_000 -> '₹15.00 L' (or 'INR 15.00 L')."""
    symbol = "INR " if ascii_only else config.CURRENCY_SYMBOL
    return f"{symbol}{float(value) / 1e5:,.2f} L"


def mwh(kwh: float) -> str:
    """Format energy in MWh, e.g. 164386 -> '164.4 MWh'."""
    return f"{float(kwh) / 1000.0:,.1f} MWh"


def tonnes(value: float) -> str:
    """Format tonnes, e.g. 1234.5 -> '1,235 t'."""
    return f"{float(value):,.0f} t"


def pct(fraction: float) -> str:
    """Format a fraction as a percentage, e.g. 0.083 -> '8.3%'."""
    return f"{float(fraction) * 100:.1f}%"


def na_or(value: object, fmt: str = "{:.3f}") -> str:
    """Return 'n/a' for None/NaN, else the formatted value."""
    if value is None:
        return "n/a"
    try:
        if isinstance(value, float) and math.isnan(value):
            return "n/a"
    except (TypeError, ValueError):
        pass
    try:
        return fmt.format(value)
    except (ValueError, TypeError):
        return str(value)


def coefficient_equation(coefficients: Mapping[str, float], target: str = config.TARGET_COLUMN) -> str:
    """Render a fitted model as an equation, constant first.

    e.g. {'const': 181.5, 'production_tonnes': 66.7} ->
    'energy_kwh = 181.5 + 66.7·production_tonnes'.
    """
    const = float(coefficients.get("const", 0.0))
    terms = [f"{const:.1f}"]
    for name, coef in coefficients.items():
        if name == "const":
            continue
        sign = "+" if coef >= 0 else "-"
        terms.append(f"{sign} {abs(float(coef)):.1f}·{name}")
    return f"{target} = " + " ".join(terms)
