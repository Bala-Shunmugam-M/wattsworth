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
