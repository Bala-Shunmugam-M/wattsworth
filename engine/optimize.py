"""Optimisation: ranked energy-saving actions with INR + CO2 + payback.

Three optimisers, each returning :class:`OptimizationAction` items with annual
₹ saving, CO2 avoided, capex, and simple payback:

* :func:`motor_right_sizing` — under-loaded motors run at poor efficiency; a
  smaller motor at ~75% load draws less. Energy + ₹ + CO2 saving.
* :func:`time_of_use_shifting` — moving load from peak to off-peak doesn't cut
  kWh, but cuts the bill via the tariff spread. ₹ saving only.
* :func:`power_factor_correction` — capacitors raise power factor, cutting the
  apparent-power (kVA) demand charge. ₹ saving only.

Pure Python — no Streamlit imports.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from engine import carbon, config, economics


@dataclass
class OptimizationAction:
    """A single recommended action.

    Attributes:
        action: Human-readable recommendation.
        category: One of ``'motor'``, ``'load_shift'``, ``'power_factor'``.
        kwh_saved: Annual energy saving (kWh). Zero for purely financial actions.
        inr_saved: Annual monetary saving (INR).
        tonnes_co2_avoided: Annual CO2 avoided (tonnes).
        payback_months: Simple payback period (0 if no capex; inf if no saving).
        capex_inr: Up-front investment required (INR).
        asset_id: Related asset identifier, if any.
    """

    action: str
    category: str
    kwh_saved: float = 0.0
    inr_saved: float = 0.0
    tonnes_co2_avoided: float = 0.0
    payback_months: float = float("inf")
    capex_inr: float = 0.0
    asset_id: str = ""


def motor_efficiency(load_factor: float) -> float:
    """Return motor efficiency (fraction) for a given load factor.

    A simple parabola peaking at the ideal load (~75%): efficiency sags at low
    load and, mildly, under overload. Calibrated to ~0.93 peak, ~0.87 at 30%.

    Args:
        load_factor: Mechanical load as a fraction of rating.

    Returns:
        Efficiency in the range [0.70, 0.95].
    """
    lf = min(max(load_factor, 0.05), 1.20)
    eff = 0.93 - 0.30 * (lf - config.MOTOR_TARGET_LOAD) ** 2
    return float(min(max(eff, 0.70), 0.95))


def motor_right_sizing(
    motors: pd.DataFrame,
    tariff_inr_per_kwh: float | None = None,
    emission_factor: float | None = None,
) -> list[OptimizationAction]:
    """Recommend right-sizing for under-loaded motors.

    For each motor below :data:`config.UNDERLOADED_LF`, recommends a motor sized so
    the same shaft load sits at the ideal ~75% load factor, and estimates the
    energy saving from the resulting efficiency gain:

        shaft power  = current input × current efficiency
        new input    = shaft power / new efficiency
        power saved  = current input − new input

    Args:
        motors: Asset register with ``measured_kw``, ``load_factor``,
            ``run_hours_per_day`` (``rated_kw``, ``name``, ``asset_id`` optional).
        tariff_inr_per_kwh: Energy price; defaults to the flat configured tariff.
        emission_factor: kg CO2 per kWh; defaults to the configured grid factor.

    Returns:
        A list of motor right-sizing actions (one per under-loaded motor).
    """
    tariff = tariff_inr_per_kwh if tariff_inr_per_kwh is not None else config.TARIFF_FLAT_INR_PER_KWH
    factor = emission_factor if emission_factor is not None else config.GRID_EMISSION_FACTOR_KG_PER_KWH

    actions: list[OptimizationAction] = []
    for _, m in motors.iterrows():
        lf = float(m["load_factor"])
        if lf >= config.UNDERLOADED_LF:
            continue

        measured_kw = float(m["measured_kw"])
        run_hours = float(m.get("run_hours_per_day", 20.0))
        eff_old = motor_efficiency(lf)
        eff_new = motor_efficiency(config.MOTOR_TARGET_LOAD)

        power_saved_kw = measured_kw * (1.0 - eff_old / eff_new)
        annual_kwh = power_saved_kw * run_hours * 365.0
        if annual_kwh <= 0:
            continue

        new_rated_kw = measured_kw / config.MOTOR_TARGET_LOAD
        capex = new_rated_kw * config.MOTOR_COST_INR_PER_KW
        inr = economics.energy_cost(annual_kwh, tariff)
        payback = economics.simple_payback_months(capex, inr / 12.0)

        name = str(m.get("name", m.get("asset_id", "motor")))
        rated = float(m.get("rated_kw", new_rated_kw))
        actions.append(
            OptimizationAction(
                action=(
                    f"Right-size {name}: {rated:.0f} kW → {new_rated_kw:.0f} kW "
                    f"(running at {lf:.0%} load, η {eff_old:.0%}→{eff_new:.0%})."
                ),
                category="motor",
                kwh_saved=annual_kwh,
                inr_saved=inr,
                tonnes_co2_avoided=carbon.co2_emissions_tonnes(annual_kwh, factor),
                payback_months=payback,
                capex_inr=capex,
                asset_id=str(m.get("asset_id", "")),
            )
        )
    return actions


def time_of_use_shifting(
    df: pd.DataFrame,
    shiftable_fraction: float = config.DEFAULT_SHIFTABLE_FRACTION,
    tariff_peak: float | None = None,
    tariff_offpeak: float | None = None,
    target: str = config.TARGET_COLUMN,
) -> list[OptimizationAction]:
    """Recommend shifting load from peak to off-peak.

    Shifting does not reduce kWh, so there is no energy or CO2 saving — only a
    bill reduction from the peak/off-peak tariff spread.

    Args:
        df: Plant-energy frame with ``target``.
        shiftable_fraction: Fraction of annual energy that can be shifted.
        tariff_peak: Peak tariff; defaults to configured peak.
        tariff_offpeak: Off-peak tariff; defaults to configured off-peak.
        target: Energy column.

    Returns:
        A single-element list (or empty if no positive saving).

    Raises:
        ValueError: If ``target`` is missing or ``shiftable_fraction`` is invalid.
    """
    if target not in df.columns:
        raise ValueError(f"Missing energy column '{target}'.")
    if not 0.0 <= shiftable_fraction <= 1.0:
        raise ValueError("shiftable_fraction must be between 0 and 1.")

    peak = tariff_peak if tariff_peak is not None else config.TARIFF_PEAK_INR_PER_KWH
    offpeak = tariff_offpeak if tariff_offpeak is not None else config.TARIFF_OFFPEAK_INR_PER_KWH
    spread = peak - offpeak
    if spread <= 0:
        return []

    annual_energy = float(df[target].mean()) * 365.0
    shiftable_kwh = annual_energy * shiftable_fraction
    inr = shiftable_kwh * spread
    if inr <= 0:
        return []

    return [
        OptimizationAction(
            action=(
                f"Shift {shiftable_fraction:.0%} of load to off-peak "
                f"({shiftable_kwh / 1000:,.0f} MWh/yr at ₹{spread:.1f}/kWh spread)."
            ),
            category="load_shift",
            kwh_saved=0.0,
            inr_saved=inr,
            tonnes_co2_avoided=0.0,
            payback_months=0.0,
            capex_inr=0.0,
        )
    ]


def power_factor_correction(
    motors: pd.DataFrame,
    target_pf: float = config.TARGET_POWER_FACTOR,
) -> list[OptimizationAction]:
    """Recommend capacitor correction to raise plant power factor.

    Sizes capacitors (kVAR) to bring motors below ``target_pf`` up to target, and
    values the saving as the reduced apparent-power (kVA) demand charge:

        kVAR needed = Σ P·(tanφ_now − tanφ_target)
        kVA saved   = kVA_now − P_total / target_pf
        ₹/yr        = kVA saved × demand charge × 12

    Args:
        motors: Asset register with ``measured_kw`` and ``power_factor``.
        target_pf: Target power factor (e.g. 0.95).

    Returns:
        A single-element list (or empty if power factor is already at/above target).
    """
    low = motors[motors["power_factor"] < target_pf]
    if low.empty:
        return []

    p_total = float(low["measured_kw"].astype(float).sum())
    phi_target = math.acos(target_pf)
    tan_target = math.tan(phi_target)

    q_now = 0.0
    for _, m in low.iterrows():
        p = float(m["measured_kw"])
        pf = min(max(float(m["power_factor"]), 0.05), 0.999)
        q_now += p * math.tan(math.acos(pf))

    q_target = p_total * tan_target
    kvar_needed = max(q_now - q_target, 0.0)
    if kvar_needed <= 0:
        return []

    kva_now = math.hypot(p_total, q_now)
    kva_new = math.hypot(p_total, q_target)  # == p_total / target_pf
    kva_saved = max(kva_now - kva_new, 0.0)

    annual_inr = kva_saved * config.DEMAND_CHARGE_INR_PER_KVA_MONTH * 12.0
    capex = kvar_needed * config.CAPACITOR_COST_INR_PER_KVAR
    payback = economics.simple_payback_months(capex, annual_inr / 12.0)
    pf_now = p_total / kva_now if kva_now else target_pf

    return [
        OptimizationAction(
            action=(
                f"Install {kvar_needed:,.0f} kVAR of PF correction "
                f"(plant PF {pf_now:.2f} → {target_pf:.2f}, saves {kva_saved:,.0f} kVA demand)."
            ),
            category="power_factor",
            kwh_saved=0.0,
            inr_saved=annual_inr,
            tonnes_co2_avoided=0.0,
            payback_months=payback,
            capex_inr=capex,
        )
    ]


def rank_actions(actions: list[OptimizationAction]) -> list[OptimizationAction]:
    """Return ``actions`` sorted by INR saved, descending."""
    return sorted(actions, key=lambda a: a.inr_saved, reverse=True)


def recommend_all(
    motors: pd.DataFrame,
    df: pd.DataFrame,
    tariff_inr_per_kwh: float | None = None,
    emission_factor: float | None = None,
    shiftable_fraction: float = config.DEFAULT_SHIFTABLE_FRACTION,
    target_pf: float = config.TARGET_POWER_FACTOR,
) -> list[OptimizationAction]:
    """Run all optimisers and return a single list ranked by INR saved.

    Args:
        motors: Asset register.
        df: Plant-energy frame.
        tariff_inr_per_kwh: Energy price for motor savings; defaults to flat tariff.
        emission_factor: kg CO2 per kWh; defaults to configured grid factor.
        shiftable_fraction: Fraction of load shiftable to off-peak.
        target_pf: Target power factor.

    Returns:
        All recommended actions, ranked by annual ₹ saving.
    """
    actions: list[OptimizationAction] = []
    actions += motor_right_sizing(motors, tariff_inr_per_kwh, emission_factor)
    actions += time_of_use_shifting(df, shiftable_fraction)
    actions += power_factor_correction(motors, target_pf)
    return rank_actions(actions)
