"""Tests for the optimisation engine (engine/optimize.py)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import config, data, optimize  # noqa: E402


@pytest.fixture(scope="module")
def motors() -> pd.DataFrame:
    return data.generate_motor_register(seed=42)


@pytest.fixture(scope="module")
def plant() -> pd.DataFrame:
    return data.generate_synthetic_plant_data(days=540, seed=42)


# --- Efficiency curve --------------------------------------------------------
def test_efficiency_peaks_near_target() -> None:
    assert optimize.motor_efficiency(0.75) > optimize.motor_efficiency(0.30)
    assert optimize.motor_efficiency(0.75) > optimize.motor_efficiency(1.10)
    assert 0.70 <= optimize.motor_efficiency(0.10) <= 0.95


# --- Motor right-sizing ------------------------------------------------------
def test_identifies_three_underloaded_motors(motors: pd.DataFrame) -> None:
    actions = optimize.motor_right_sizing(motors)
    assert len(actions) == 3
    flagged = {a.asset_id for a in actions}
    expected = set(motors.loc[motors["load_factor"] < config.UNDERLOADED_LF, "asset_id"])
    assert flagged == expected


def test_motor_savings_are_realistic(motors: pd.DataFrame) -> None:
    for a in optimize.motor_right_sizing(motors):
        assert a.category == "motor"
        assert a.kwh_saved > 0
        assert a.inr_saved > 0
        assert a.tonnes_co2_avoided > 0
        assert a.capex_inr > 0
        assert 0 < a.payback_months < 240          # under 20 years
        assert a.kwh_saved < 500_000               # sane annual magnitude


def test_motor_payback_consistent(motors: pd.DataFrame) -> None:
    """Payback equals capex / monthly saving."""
    for a in optimize.motor_right_sizing(motors):
        expected = a.capex_inr / (a.inr_saved / 12.0)
        assert a.payback_months == pytest.approx(expected, rel=1e-6)


def test_money_and_carbon_follow_energy(motors: pd.DataFrame) -> None:
    actions = optimize.motor_right_sizing(motors, tariff_inr_per_kwh=8.0, emission_factor=0.71)
    for a in actions:
        assert a.inr_saved == pytest.approx(a.kwh_saved * 8.0)
        assert a.tonnes_co2_avoided == pytest.approx(a.kwh_saved * 0.71 / 1000.0)


# --- Time-of-use shifting ----------------------------------------------------
def test_tou_saves_money_not_energy(plant: pd.DataFrame) -> None:
    actions = optimize.time_of_use_shifting(plant, shiftable_fraction=0.15)
    assert len(actions) == 1
    a = actions[0]
    assert a.category == "load_shift"
    assert a.inr_saved > 0
    assert a.kwh_saved == 0.0          # shifting doesn't cut energy
    assert a.payback_months == 0.0     # operational, no capex


def test_tou_scales_with_fraction(plant: pd.DataFrame) -> None:
    small = optimize.time_of_use_shifting(plant, shiftable_fraction=0.10)[0]
    big = optimize.time_of_use_shifting(plant, shiftable_fraction=0.20)[0]
    assert big.inr_saved == pytest.approx(2.0 * small.inr_saved, rel=1e-6)


def test_tou_bad_fraction_raises(plant: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        optimize.time_of_use_shifting(plant, shiftable_fraction=1.5)


def test_tou_only_shifts_peak_energy(plant: pd.DataFrame) -> None:
    """Saving is based on the actual peak-period energy, not all energy."""
    a = optimize.time_of_use_shifting(plant, shiftable_fraction=0.15)[0]
    # Naive (wrong) figure shifts ALL energy; correct figure uses only peak days.
    annual_all = plant["energy_kwh"].mean() * 365.0
    naive = annual_all * 0.15 * (config.TARIFF_PEAK_INR_PER_KWH - config.TARIFF_OFFPEAK_INR_PER_KWH)
    assert 0 < a.inr_saved < naive


# --- Power-factor correction -------------------------------------------------
def test_pf_correction_produces_action(motors: pd.DataFrame) -> None:
    actions = optimize.power_factor_correction(motors, target_pf=0.95)
    assert len(actions) == 1
    a = actions[0]
    assert a.category == "power_factor"
    assert a.inr_saved > 0
    assert a.capex_inr > 0
    assert a.payback_months > 0
    assert "kVAR" in a.action


def test_pf_no_action_when_already_good() -> None:
    good = pd.DataFrame(
        {"measured_kw": [50.0, 30.0], "power_factor": [0.97, 0.98], "asset_id": ["M-1", "M-2"]}
    )
    assert optimize.power_factor_correction(good, target_pf=0.95) == []


# --- Ranking & aggregation ---------------------------------------------------
def test_rank_actions_orders_by_inr() -> None:
    actions = [
        optimize.OptimizationAction("a", "motor", inr_saved=100.0),
        optimize.OptimizationAction("b", "motor", inr_saved=900.0),
        optimize.OptimizationAction("c", "motor", inr_saved=500.0),
    ]
    ranked = optimize.rank_actions(actions)
    assert [a.inr_saved for a in ranked] == [900.0, 500.0, 100.0]


def test_recommend_all_combines_and_ranks(motors: pd.DataFrame, plant: pd.DataFrame) -> None:
    actions = optimize.recommend_all(motors, plant)
    # 3 motors + 1 ToU + 1 PF = 5 actions
    assert len(actions) == 5
    categories = {a.category for a in actions}
    assert categories == {"motor", "load_shift", "power_factor"}
    # Ranked descending by ₹.
    savings = [a.inr_saved for a in actions]
    assert savings == sorted(savings, reverse=True)


# --- Capex-budget knapsack ---------------------------------------------------
def test_knapsack_respects_budget(motors: pd.DataFrame, plant: pd.DataFrame) -> None:
    """The selected plan never exceeds the capex budget."""
    actions = optimize.recommend_all(motors, plant)
    budget = 120_000.0
    plan = optimize.optimize_under_budget(actions, budget_inr=budget)
    assert plan.total_capex <= budget
    assert plan.total_inr == pytest.approx(sum(a.inr_saved for a in plan.selected))


def test_knapsack_is_optimal_not_just_greedy() -> None:
    """Knapsack picks the highest-value affordable subset, not the cheapest items."""
    actions = [
        optimize.OptimizationAction("cheap-low", "motor", inr_saved=10.0, capex_inr=40.0),
        optimize.OptimizationAction("mid", "motor", inr_saved=70.0, capex_inr=50.0),
        optimize.OptimizationAction("pricey-high", "motor", inr_saved=90.0, capex_inr=60.0),
    ]
    plan = optimize.optimize_under_budget(actions, budget_inr=100.0)
    # Optimal under budget 100: mid(50)+cheap(40)=90 vs pricey(60)+cheap(40)=100 -> value 70+10=80 vs 90+10=100
    assert plan.total_capex <= 100.0
    assert plan.total_inr == pytest.approx(100.0)
    assert {a.action for a in plan.selected} == {"pricey-high", "cheap-low"}


def test_knapsack_zero_budget_keeps_only_free_actions(plant: pd.DataFrame) -> None:
    """With no capex, only zero-capex actions (e.g. ToU shifting) can be selected."""
    actions = [
        optimize.OptimizationAction("free", "load_shift", inr_saved=500.0, capex_inr=0.0),
        optimize.OptimizationAction("paid", "motor", inr_saved=900.0, capex_inr=50_000.0),
    ]
    plan = optimize.optimize_under_budget(actions, budget_inr=0.0)
    assert [a.action for a in plan.selected] == ["free"]


def test_knapsack_can_maximise_co2() -> None:
    """The objective can be tCO2 avoided instead of rupees."""
    actions = [
        optimize.OptimizationAction("a", "motor", inr_saved=100.0, tonnes_co2_avoided=1.0, capex_inr=50.0),
        optimize.OptimizationAction("b", "motor", inr_saved=10.0, tonnes_co2_avoided=9.0, capex_inr=50.0),
    ]
    plan = optimize.optimize_under_budget(actions, budget_inr=50.0, objective="tonnes_co2_avoided")
    assert [a.action for a in plan.selected] == ["b"]


def test_knapsack_bad_objective_raises() -> None:
    with pytest.raises(ValueError, match="objective must be"):
        optimize.optimize_under_budget([], budget_inr=100.0, objective="payback_months")


def test_motor_sizing_uses_measured_efficiency(motors: pd.DataFrame) -> None:
    """Motor savings reflect the register's measured efficiency, not only the curve."""
    bumped = motors.copy()
    # Drop one under-loaded motor's measured efficiency sharply -> bigger saving.
    mask = bumped["load_factor"] < config.UNDERLOADED_LF
    base = optimize.motor_right_sizing(motors)
    bumped.loc[mask, "efficiency_pct"] = 70.0
    worse = optimize.motor_right_sizing(bumped)
    assert sum(a.kwh_saved for a in worse) > sum(a.kwh_saved for a in base)
