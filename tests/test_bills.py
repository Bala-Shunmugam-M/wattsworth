"""Tests for engine.bills — monthly-bill ingestion and the Bill Snapshot."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine import bills


def _raw(months, kwh, amount, pf=None, demand=None) -> pd.DataFrame:
    d = {"month": months, "energy_kwh": kwh, "amount_inr": amount}
    if pf is not None:
        d["power_factor"] = pf
    if demand is not None:
        d["max_demand_kva"] = demand
    return pd.DataFrame(d)


# --------------------------------------------------------------- validation ----

def test_clean_bills_pass_through():
    raw = _raw(["2025-01", "2025-02", "2025-03"], [10000, 11000, 9000], [80000, 88000, 72000])
    df, rep = bills.validate_and_clean_bills(raw)
    assert rep.ok
    assert rep.rows_in == 3 and rep.rows_out == 3
    assert list(df["month"]) == ["2025-01", "2025-02", "2025-03"]
    assert "period" in df.columns


def test_missing_essential_column_is_not_ok():
    raw = pd.DataFrame({"month": ["2025-01"], "energy_kwh": [1000]})  # no amount_inr
    df, rep = bills.validate_and_clean_bills(raw)
    assert not rep.ok
    assert "amount_inr" in rep.missing_columns
    assert df.empty


def test_optional_columns_filled_when_absent():
    raw = _raw(["2025-01", "2025-02"], [1000, 1100], [8000, 8800])
    df, rep = bills.validate_and_clean_bills(raw)
    assert {"power_factor", "max_demand_kva"}.issubset(df.columns)
    assert df["power_factor"].isna().all()


def test_messy_numbers_are_coerced():
    raw = _raw(["2025-01", "2025-02", "2025-03"],
               ["10,000", "11,000", "N/A"], ["80,000", "88,000", "72,000"])
    df, rep = bills.validate_and_clean_bills(raw)
    assert rep.coerced_cells >= 1
    # The N/A energy row is dropped (essential).
    assert rep.rows_out == 2


def test_duplicate_months_keep_last():
    raw = _raw(["2025-01", "2025-01", "2025-02"], [1000, 2000, 1500], [8000, 16000, 12000])
    df, rep = bills.validate_and_clean_bills(raw)
    assert rep.duplicate_rows_removed == 1
    assert rep.rows_out == 2
    jan = df[df["month"] == "2025-01"]
    assert float(jan["energy_kwh"].iloc[0]) == 2000.0  # last kept


def test_unparseable_month_dropped():
    raw = _raw(["2025-01", "not-a-date"], [1000, 1100], [8000, 8800])
    df, rep = bills.validate_and_clean_bills(raw)
    assert rep.rows_out == 1
    assert rep.dropped_invalid_rows == 1


def test_nonpositive_energy_and_negative_amount_dropped():
    raw = _raw(["2025-01", "2025-02", "2025-03"], [0, 1000, 1100], [8000, -5, 8800])
    df, rep = bills.validate_and_clean_bills(raw)
    assert rep.rows_out == 1  # only the third row is fully valid


def test_power_factor_out_of_range_nulled():
    raw = _raw(["2025-01", "2025-02"], [1000, 1100], [8000, 8800], pf=[1.4, 0.9])
    df, rep = bills.validate_and_clean_bills(raw)
    assert pd.isna(df["power_factor"].iloc[0])
    assert float(df["power_factor"].iloc[1]) == 0.9


def test_few_bills_warns_but_ok():
    raw = _raw(["2025-01"], [1000], [8000])
    df, rep = bills.validate_and_clean_bills(raw)
    assert rep.ok
    assert any("3+" in m for m in rep.messages)


def test_empty_frame_not_ok():
    df, rep = bills.validate_and_clean_bills(pd.DataFrame({"month": [], "energy_kwh": [], "amount_inr": []}))
    assert not rep.ok


# ------------------------------------------------------------------ snapshot ----

def _clean(*args, **kwargs):
    df, rep = bills.validate_and_clean_bills(_raw(*args, **kwargs))
    assert rep.ok
    return df


def test_snapshot_totals_and_rate():
    df = _clean(["2025-01", "2025-02", "2025-03"], [10000, 10000, 10000], [80000, 80000, 80000])
    s = bills.compute_snapshot(df)
    assert s.n_bills == 3
    assert s.total_kwh == 30000
    assert s.total_inr == 240000
    assert s.avg_monthly_inr == 80000
    assert s.effective_rate_inr_per_kwh == pytest.approx(8.0)


def test_snapshot_highest_lowest_month():
    df = _clean(["2025-01", "2025-02", "2025-03"], [9000, 12000, 8000], [72000, 96000, 64000])
    s = bills.compute_snapshot(df)
    assert s.highest_month == "2025-02"
    assert s.lowest_month == "2025-03"


def test_spend_observation_always_present():
    df = _clean(["2025-01", "2025-02", "2025-03"], [10000, 10000, 10000], [80000, 80000, 80000])
    s = bills.compute_snapshot(df)
    assert any(o.code == "SPEND" and o.confidence == "High" for o in s.observations)


def test_rate_volatility_flagged():
    # Same kWh, very different ₹ → effective rate swings.
    df = _clean(["2025-01", "2025-02", "2025-03"], [10000, 10000, 10000], [60000, 100000, 80000])
    s = bills.compute_snapshot(df)
    assert any(o.code == "RATE_VOL" for o in s.observations)


def test_consumption_spike_flagged():
    df = _clean(["2025-01", "2025-02", "2025-03", "2025-04"],
                [10000, 10000, 20000, 10000], [80000, 80000, 160000, 80000])
    s = bills.compute_snapshot(df)
    assert any(o.code == "SPIKE" for o in s.observations)


def test_low_power_factor_gives_indicative_penalty():
    df = _clean(["2025-01", "2025-02", "2025-03"], [10000, 10000, 10000],
                [80000, 80000, 80000], pf=[0.85, 0.85, 0.85])
    s = bills.compute_snapshot(df)
    pf = [o for o in s.observations if o.code == "PF"]
    assert pf and pf[0].indicative_inr_per_year is not None
    assert pf[0].indicative_inr_per_year > 0


def test_good_power_factor_no_penalty_observation():
    df = _clean(["2025-01", "2025-02", "2025-03"], [10000, 10000, 10000],
                [80000, 80000, 80000], pf=[0.98, 0.98, 0.98])
    s = bills.compute_snapshot(df)
    assert not any(o.code == "PF" for o in s.observations)


def test_missing_power_factor_prompts_to_ask():
    df = _clean(["2025-01", "2025-02", "2025-03"], [10000, 10000, 10000], [80000, 80000, 80000])
    s = bills.compute_snapshot(df)
    assert any(o.code == "PF_UNKNOWN" for o in s.observations)


def test_empty_snapshot_raises():
    empty = pd.DataFrame(columns=[*bills.BILL_COLUMNS, "period"])
    with pytest.raises(ValueError):
        bills.compute_snapshot(empty)


# ------------------------------------------------------------------ markdown ----

def test_markdown_has_disclaimers_and_real_numbers():
    df = _clean(["2025-01", "2025-02", "2025-03"], [10000, 10000, 10000], [80000, 80000, 80000])
    s = bills.compute_snapshot(df)
    md = bills.render_snapshot_markdown(s, company="Shrivik", date="2026-06-07")
    assert "Shrivik" in md
    assert "not** an audit" in md
    assert "exploratory discussion document" in md
    assert "240,000" in md  # total spend, real number


def test_markdown_no_indicative_line_without_pf():
    df = _clean(["2025-01", "2025-02", "2025-03"], [10000, 10000, 10000], [80000, 80000, 80000])
    s = bills.compute_snapshot(df)
    md = bills.render_snapshot_markdown(s)
    assert "Indicative impact" not in md  # nothing fabricated when PF absent


def test_markdown_indicative_line_present_with_low_pf():
    df = _clean(["2025-01", "2025-02", "2025-03"], [10000, 10000, 10000],
                [80000, 80000, 80000], pf=[0.85, 0.85, 0.85])
    s = bills.compute_snapshot(df)
    md = bills.render_snapshot_markdown(s)
    assert "Indicative impact (confirm)" in md
