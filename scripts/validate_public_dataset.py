"""Validate the WattsWorth ISO 50001 baseline engine on a REAL public dataset.

Dataset: UCI / Mendeley "Steel Industry Energy Consumption" — 35,040 fifteen-minute
records from DAEWOO Steel Co. Ltd (Gwangyang, South Korea), 2018. This is genuine
industrial steel-plant data that WattsWorth never generated, so it answers the
sharpest objection to a synthetic-data project: "what does your baseline do on data
you didn't make?"

The engine's `fit_baseline` accepts arbitrary driver columns, so we aggregate the
15-minute series to daily and regress daily energy on operational drivers derived
from the data (load intensity, active hours, reactive power, power factor, weekend).

Run:  python scripts/validate_public_dataset.py
It reads data/external/steel/Steel_industry_data.csv if present (downloaded), writes
a small committed daily aggregate to data/steel_industry_daily.csv, and prints a
report. If only the daily aggregate exists, it uses that (no download needed).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import baseline  # noqa: E402

RAW = ROOT / "data" / "external" / "steel" / "Steel_industry_data.csv"
DAILY = ROOT / "data" / "steel_industry_daily.csv"

_LOAD_MAP = {"Light_Load": 1.0, "Medium_Load": 2.0, "Maximum_Load": 3.0}
# Independent, non-collinear ISO 50001 relevant variables: production intensity
# (load class) and a calendar effect. Reactive power is co-determined with energy
# and active_hours is collinear with load_index, so both are excluded.
_DRIVERS = ["load_index", "weekend"]


def build_daily() -> pd.DataFrame:
    """Aggregate the raw 15-minute steel data to a daily ISO 50001-style frame."""
    df = pd.read_csv(RAW)
    df["dt"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M")
    df["day"] = df["dt"].dt.normalize()
    df["load_num"] = df["Load_Type"].map(_LOAD_MAP)
    df["active"] = (df["Load_Type"] != "Light_Load").astype(int)

    daily = df.groupby("day").agg(
        energy_kwh=("Usage_kWh", "sum"),
        reactive_kvarh=("Lagging_Current_Reactive.Power_kVarh", "sum"),
        load_index=("load_num", "mean"),
        active_hours=("active", lambda s: float(s.sum()) * 0.25),
        mean_pf=("Lagging_Current_Power_Factor", "mean"),
        weekend=("WeekStatus", lambda s: float((s == "Weekend").mean() > 0.5)),
    ).reset_index().rename(columns={"day": "date"})
    return daily


def main() -> None:
    if RAW.exists():
        daily = build_daily()
        daily.to_csv(DAILY, index=False)
        source = f"raw 15-min file ({RAW.name}) aggregated to daily"
    elif DAILY.exists():
        daily = pd.read_csv(DAILY, parse_dates=["date"])
        source = f"committed daily aggregate ({DAILY.name})"
    else:
        raise SystemExit("No data found. Download the UCI steel dataset to data/external/steel/ first.")

    model = baseline.fit_baseline(daily, target="energy_kwh", drivers=_DRIVERS)

    print("=" * 70)
    print("WattsWorth baseline — validation on REAL public data")
    print("Source: UCI/Mendeley Steel Industry Energy Consumption (DAEWOO Steel, KR)")
    print(f"Loaded: {source}")
    print("=" * 70)
    print(f"Daily observations : {len(daily)}")
    print(f"Drivers offered     : {_DRIVERS}")
    print(f"Drivers retained    : {model.drivers}")
    print(f"Drivers dropped     : {model.dropped or 'none'}")
    print("-" * 70)
    print(f"R^2                 : {model.r_squared:.3f}")
    print(f"CV(RMSE) in-sample  : {model.cv_rmse * 100:.1f}%")
    oos = "n/a" if model.cv_rmse_oos is None else f"{model.cv_rmse_oos * 100:.1f}%"
    print(f"CV(RMSE) out-of-samp: {oos}")
    print(f"Durbin-Watson       : {model.durbin_watson:.2f}  (~2 = no autocorrelation)")
    mv = "n/a" if np.isnan(model.max_vif) else f"{model.max_vif:.1f}"
    print(f"Max VIF             : {mv}  (>10 = multicollinearity)")
    bp = "n/a" if np.isnan(model.bp_pvalue) else f"{model.bp_pvalue:.3f}"
    print(f"Breusch-Pagan p     : {bp}  (<0.05 = heteroskedastic)")
    print("-" * 70)
    print("Coefficients:")
    for term, coef in model.coefficients.items():
        print(f"  {term:<16} {coef:>14.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
