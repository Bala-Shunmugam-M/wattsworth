"""Data layer: synthetic plant-energy generation, asset register, and loaders.

This module owns the WattsWorth data schema and produces a *credible* synthetic
dataset for a 100 TPD coal-based DRI / sponge-iron plant:

* ~18 months of daily site-electrical data with a realistic driver relationship
  (energy depends on production, ambient temperature, and operating hours);
* a genuine **6% step reduction in energy intensity** on 2025-09-01, simulating
  an energy-efficiency project, so IPMVP Option C M&V has a real saving to find;
* **5 injected anomaly days** (+15%..+25% spikes) for anomaly detection to catch.

The generator records ground truth (intervention date and anomaly dates) on the
returned frame's ``.attrs`` so tests can assert against the exact injected days.
``.attrs`` is in-memory only and is intentionally *not* persisted to CSV — the
analytics engine must rediscover these events from the data, never read them.

Loaders return ``None`` when a file is absent so the UI can degrade gracefully
and offer to generate sample data. Pure Python — no Streamlit imports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from engine import config

# --- Schema ------------------------------------------------------------------
PLANT_ENERGY_COLUMNS: list[str] = [
    "date",
    "energy_kwh",
    "production_tonnes",
    "ambient_temp_c",
    "operating_hours",
    "grid_pf",
    "tariff_period",
]
MOTOR_COLUMNS: list[str] = [
    "asset_id",
    "name",
    "rated_kw",
    "measured_kw",
    "load_factor",
    "efficiency_pct",
    "power_factor",
    "current_imbalance_pct",
    "run_hours_per_day",
    "criticality",
]

# --- Ground-truth events (kept stable for reproducibility & testing) ---------
INTERVENTION_DATE: pd.Timestamp = pd.Timestamp("2025-09-01")
STEP_REDUCTION: float = 0.06          # 6% drop in energy intensity post-intervention
N_ANOMALIES: int = 5                  # number of injected spike days
ANOMALY_MIN_MULT: float = 1.15        # +15%
ANOMALY_MAX_MULT: float = 1.25        # +25%

# --- Plant physics-ish coefficients (tuned for a 100 TPD DRI plant) ----------
_FIXED_LOAD_KWH: float = 1500.0       # always-on auxiliaries (lighting, instrumentation)
_PROD_COEF_KWH_PER_T: float = 72.0    # variable electrical per tonne of sponge iron
_TEMP_REF_C: float = 30.0             # reference ambient temperature
_TEMP_COEF_KWH_PER_C: float = 38.0    # cooling/fan load per °C above reference
_HOURS_COEF_KWH_PER_H: float = 95.0   # energy per operating hour
_NOISE_FRAC: float = 0.015            # day-to-day noise as fraction of expected energy
#   Kept low so the injected +15-25% anomaly spikes are cleanly separable from
#   natural daily variation. Robust z-scores are scale-invariant, so this lifts
#   each spike's z well above the ~3.1 background without touching spike sizes.


def generate_synthetic_plant_data(days: int = 540, seed: int = 42) -> pd.DataFrame:
    """Generate a credible synthetic daily plant-energy dataset.

    Models site electrical energy for a 100 TPD coal-based DRI plant as a linear
    function of production, ambient temperature, and operating hours, plus
    Gaussian noise. A 6% energy-intensity step reduction is injected on
    :data:`INTERVENTION_DATE`, and :data:`N_ANOMALIES` spike days are injected
    for anomaly detection.

    The date range is anchored around the intervention (half the days before,
    half after) so the event is always mid-series regardless of the run date.

    Args:
        days: Number of consecutive daily records to generate (default ~18 months).
        seed: RNG seed for reproducibility.

    Returns:
        DataFrame with columns :data:`PLANT_ENERGY_COLUMNS`. The frame's
        ``.attrs`` carries ``"intervention_date"`` (ISO string) and
        ``"anomaly_dates"`` (list of :class:`pandas.Timestamp`) as ground truth.
    """
    rng = np.random.default_rng(seed)

    # Anchor the window around the intervention so it never drifts out of range.
    start = INTERVENTION_DATE - pd.Timedelta(days=days // 2)
    dates = pd.date_range(start=start, periods=days, freq="D")
    day_of_year = dates.dayofyear.to_numpy()

    # --- Drivers -------------------------------------------------------------
    production = rng.normal(92.0, 8.0, days).clip(58.0, 106.0)          # TPD
    seasonal = 6.0 * np.sin(2.0 * np.pi * (day_of_year - 110) / 365.0)  # peak ~ summer
    ambient = (30.0 + seasonal + rng.normal(0.0, 2.0, days)).clip(20.0, 42.0)
    hours = rng.normal(22.5, 1.1, days).clip(14.0, 24.0)               # near-continuous kiln
    grid_pf = rng.normal(0.84, 0.04, days).clip(0.72, 0.97)
    tariff_period = rng.choice(["peak", "offpeak"], size=days, p=[0.55, 0.45])

    # --- Expected (deterministic) energy ------------------------------------
    expected = (
        _FIXED_LOAD_KWH
        + _PROD_COEF_KWH_PER_T * production
        + _TEMP_COEF_KWH_PER_C * (ambient - _TEMP_REF_C)
        + _HOURS_COEF_KWH_PER_H * hours
    )

    # 6% intensity step reduction from the intervention date onward.
    post = np.asarray(dates >= INTERVENTION_DATE)
    expected = expected * np.where(post, 1.0 - STEP_REDUCTION, 1.0)

    # Realistic day-to-day noise proportional to the expected level.
    energy = expected + rng.normal(0.0, _NOISE_FRAC * expected, days)
    energy = energy.clip(min=0.0)

    # --- Inject anomaly spikes ----------------------------------------------
    candidate_idx = np.arange(10, days - 10)
    anomaly_idx = np.sort(rng.choice(candidate_idx, size=N_ANOMALIES, replace=False))
    spike_mult = rng.uniform(ANOMALY_MIN_MULT, ANOMALY_MAX_MULT, N_ANOMALIES)
    energy[anomaly_idx] = energy[anomaly_idx] * spike_mult

    df = pd.DataFrame(
        {
            "date": dates,
            "energy_kwh": energy.round(0),
            "production_tonnes": production.round(1),
            "ambient_temp_c": ambient.round(1),
            "operating_hours": hours.round(1),
            "grid_pf": grid_pf.round(3),
            "tariff_period": tariff_period,
        }
    )

    # Ground truth for testing / debugging (in-memory only; not saved to CSV).
    df.attrs["intervention_date"] = INTERVENTION_DATE.isoformat()
    df.attrs["anomaly_dates"] = [pd.Timestamp(d) for d in dates[anomaly_idx]]
    return df


def generate_motor_register(seed: int = 42) -> pd.DataFrame:
    """Generate a credible motor asset register for a DRI plant.

    Hand-tuned so optimisation has clear, deterministic targets:

    * **3 under-loaded** motors (``load_factor < 0.40``) — right-sizing candidates;
    * **2 overloaded** motors (``load_factor > 1.0``) — thermal/reliability risk;
    * **varied criticality** spanning A, B, and C.

    Efficiency, power factor, current imbalance, and run-hours carry light
    seeded noise for realism; load factors and criticality are fixed.

    Args:
        seed: RNG seed for the cosmetic-noise fields.

    Returns:
        DataFrame with columns :data:`MOTOR_COLUMNS`.
    """
    rng = np.random.default_rng(seed)

    names = [
        "BFW Pump 1", "BFW Pump 2", "ID Fan", "FD Fan", "Cooling Water Pump",
        "Kiln Main Drive", "Product Conveyor", "Air Compressor", "DM Water Pump",
        "Dust Extraction Fan",
    ]
    rated_kw = np.array([75, 55, 110, 90, 37, 160, 30, 45, 22, 75], dtype=float)
    # 3 under-loaded (<0.40): idx 1, 4, 8 ; 2 overloaded (>1.0): idx 2, 5
    load_factor = np.array([0.62, 0.34, 1.06, 0.88, 0.31, 1.09, 0.70, 0.80, 0.37, 0.66])
    criticality = ["A", "B", "A", "B", "C", "A", "C", "B", "C", "B"]

    measured_kw = (rated_kw * load_factor).round(1)

    rows: list[dict[str, object]] = []
    for i, name in enumerate(names):
        lf = float(load_factor[i])
        # Efficiency sags at low load and under overload; peaks near 75-90% load.
        eff = 92.5 - 26.0 * (lf - 0.80) ** 2 + float(rng.normal(0.0, 0.6))
        rows.append(
            {
                "asset_id": f"M-{i + 1:02d}",
                "name": name,
                "rated_kw": float(rated_kw[i]),
                "measured_kw": float(measured_kw[i]),
                "load_factor": round(lf, 2),
                "efficiency_pct": round(float(np.clip(eff, 78.0, 95.0)), 1),
                "power_factor": round(float(np.clip(rng.normal(0.80, 0.05), 0.6, 0.99)), 2),
                "current_imbalance_pct": round(float(abs(rng.normal(3.0, 1.5))), 1),
                "run_hours_per_day": round(float(np.clip(rng.normal(21.0, 2.0), 8.0, 24.0)), 1),
                "criticality": criticality[i],
            }
        )
    return pd.DataFrame(rows, columns=MOTOR_COLUMNS)


def save_synthetic_data(data_dir: Optional[Path] = None) -> tuple[Path, Path]:
    """Generate and persist both synthetic datasets to CSV.

    Uses the upgraded generators. Note that ground-truth ``.attrs`` are not
    written to CSV by design — downstream analytics must rediscover the
    intervention and anomalies from the data itself.

    Args:
        data_dir: Target directory; defaults to :data:`config.DATA_DIR`.

    Returns:
        Tuple of ``(plant_energy_path, motors_path)``.
    """
    data_dir = data_dir or config.DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    plant_path = data_dir / config.PLANT_ENERGY_CSV.name
    motors_path = data_dir / config.MOTORS_CSV.name
    generate_synthetic_plant_data().to_csv(plant_path, index=False)
    generate_motor_register().to_csv(motors_path, index=False)
    return plant_path, motors_path


def load_plant_energy(path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Load the plant-energy time series, or ``None`` if the file is absent.

    Args:
        path: CSV path; defaults to :data:`config.PLANT_ENERGY_CSV`.

    Returns:
        DataFrame with ``date`` parsed as datetime, or ``None``.
    """
    path = path or config.PLANT_ENERGY_CSV
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["date"])


def load_motors(path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Load the motor asset register, or ``None`` if the file is absent.

    Args:
        path: CSV path; defaults to :data:`config.MOTORS_CSV`.

    Returns:
        DataFrame, or ``None``.
    """
    path = path or config.MOTORS_CSV
    if not path.exists():
        return None
    return pd.read_csv(path)


def validate_plant_energy(df: pd.DataFrame) -> bool:
    """Return ``True`` if ``df`` contains all required plant-energy columns."""
    return set(PLANT_ENERGY_COLUMNS).issubset(df.columns)
