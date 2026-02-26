"""
generate_synthetic_data.py
--------------------------
Generate two synthetic files that mimic the real-world data setup:

1. **Production CSV**: timestamp + power_kw (simulates the user's PV panel data)
2. **SMHI weather CSV**: timestamp + temperature + ghi (simulates SMHI API cache)

The synthetic data follows Nordic seasonal patterns at ~59°N latitude.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(
    start: str = "2023-01-01",
    end: str = "2024-12-31",
    freq: str = "1h",
    production_output: str = "data/synthetic_production.csv",
    smhi_output: str = "data/synthetic_smhi.csv",
) -> None:
    """Generate and save synthetic hourly PV + weather data.

    Parameters
    ----------
    start, end : str
        Date range.
    freq : str
        Time resolution.
    production_output : str
        Output path for the production CSV.
    smhi_output : str
        Output path for the SMHI weather CSV.
    """
    rng = np.random.default_rng(42)
    timestamps = pd.date_range(start, end, freq=freq, tz="UTC")
    n = len(timestamps)

    hour = timestamps.hour + timestamps.minute / 60.0
    day_of_year = timestamps.dayofyear.astype(float)

    # ----- GHI: seasonal + diurnal pattern ----------------------------------
    seasonal_amplitude = 400 + 500 * np.sin(
        2 * np.pi * (day_of_year - 80) / 365.0
    )
    seasonal_amplitude = np.clip(seasonal_amplitude, 50, 900)

    # Diurnal bell curve
    sunrise = 9 - 5 * np.sin(2 * np.pi * (day_of_year - 80) / 365.0)
    sunrise = np.clip(sunrise, 3, 10)
    sunset = 15 + 5 * np.sin(2 * np.pi * (day_of_year - 80) / 365.0)
    sunset = np.clip(sunset, 14, 22)

    day_frac = np.where(
        (hour >= sunrise) & (hour <= sunset),
        np.sin(np.pi * (hour - sunrise) / (sunset - sunrise + 1e-6)),
        0.0,
    )

    ghi = seasonal_amplitude * day_frac
    ghi += rng.normal(0, 30, n)
    ghi = np.clip(ghi, 0, 1100)

    # ----- Temperature: seasonal + diurnal ----------------------------------
    temp_base = 5 + 15 * np.sin(2 * np.pi * (day_of_year - 100) / 365.0)
    temp_diurnal = 3 * np.sin(2 * np.pi * (hour - 6) / 24.0)
    temperature = temp_base + temp_diurnal + rng.normal(0, 2, n)

    # ----- Power output (kW) -----------------------------------------------
    gamma = -0.004
    t_stc = 25.0
    eta_t = 1 + gamma * (temperature - t_stc)
    power_kw = 10.0 * (ghi / 1000.0) * eta_t
    power_kw += rng.normal(0, 0.15, n)
    power_kw = np.clip(power_kw, 0, 12)

    # ----- Save production CSV (timestamp + power_kw) -----------------------
    prod_df = pd.DataFrame({
        "timestamp": timestamps,
        "power_kw": np.round(power_kw, 4),
    })
    prod_path = Path(production_output)
    prod_path.parent.mkdir(parents=True, exist_ok=True)
    prod_df.to_csv(prod_path, index=False)
    print(f"Production data -> {prod_path}  ({len(prod_df)} rows)")

    # ----- Save SMHI weather CSV (timestamp + temperature + ghi) ------------
    smhi_df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature": np.round(temperature, 2),
        "ghi": np.round(ghi, 2),
    })
    smhi_path = Path(smhi_output)
    smhi_path.parent.mkdir(parents=True, exist_ok=True)
    smhi_df.to_csv(smhi_path, index=False)
    print(f"SMHI weather data -> {smhi_path}  ({len(smhi_df)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic solar data")
    parser.add_argument("--production-output", "-p", default="data/synthetic_production.csv")
    parser.add_argument("--smhi-output", "-s", default="data/synthetic_smhi.csv")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args()
    generate(
        start=args.start,
        end=args.end,
        production_output=args.production_output,
        smhi_output=args.smhi_output,
    )
