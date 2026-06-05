"""Central configuration for the WattsWorth engine.

Shared constants and default parameters. Pure Python — no Streamlit and no I/O
side effects at import time.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
PLANT_ENERGY_CSV: Path = DATA_DIR / "synthetic_plant_energy.csv"
MOTORS_CSV: Path = DATA_DIR / "motors.csv"

# --- Carbon ------------------------------------------------------------------
# India grid average emission factor (kg CO2 per kWh). Editable in-app.
GRID_EMISSION_FACTOR_KG_PER_KWH: float = 0.71

# --- Economics ---------------------------------------------------------------
# Default industrial tariffs (INR per kWh).
TARIFF_PEAK_INR_PER_KWH: float = 9.5
TARIFF_OFFPEAK_INR_PER_KWH: float = 6.0
TARIFF_FLAT_INR_PER_KWH: float = 8.0
CURRENCY_SYMBOL: str = "₹"

# --- Thresholds --------------------------------------------------------------
UNDERLOADED_LF: float = 0.40        # motor load factor below this = under-loaded
OVERLOADED_LF: float = 1.00         # above this = overloaded
ANOMALY_SIGMA: float = 3.0          # residual control-limit width
TARGET_POWER_FACTOR: float = 0.95   # power-factor correction target
CVRMSE_MAX: float = 0.20            # IPMVP model acceptance: CV(RMSE) <= 20%

# --- Optimization ------------------------------------------------------------
MOTOR_TARGET_LOAD: float = 0.75              # ideal motor load factor (peak efficiency)
MOTOR_COST_INR_PER_KW: float = 4_500.0       # installed cost of a new IE3 motor (₹/kW)
CAPACITOR_COST_INR_PER_KVAR: float = 400.0   # installed cost of PF capacitors (₹/kVAR)
DEMAND_CHARGE_INR_PER_KVA_MONTH: float = 350.0  # utility demand charge (₹/kVA/month)
DEFAULT_SHIFTABLE_FRACTION: float = 0.15     # share of load that can shift to off-peak

# --- Baseline model ----------------------------------------------------------
TARGET_COLUMN: str = "energy_kwh"
DEFAULT_DRIVERS: tuple[str, ...] = (
    "production_tonnes",
    "ambient_temp_c",
    "operating_hours",
)
