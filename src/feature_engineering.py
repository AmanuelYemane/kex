"""
feature_engineering.py
----------------------
Construct thesis-relevant features from only **temperature** and **GHI**,
with emphasis on Nordic-specific phenomena: temperature-efficiency inversions,
albedo/snow effects, and seasonal irradiance variation.

No external solar geometry library (pvlib) is used. All derived features
are computed from the two raw inputs plus the timestamp.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import (
    SOLAR_CONSTANT,
    STC_TEMPERATURE,
    TARGET_COL,
    TEMP_COEFFICIENT_GAMMA,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to a cleaned DataFrame.

    The DataFrame must have a ``DatetimeIndex`` and at least the columns
    ``temperature`` and ``ghi``.

    All features are derived solely from temperature, GHI, and the
    timestamp (hour, month, day-of-year).

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``data_loader.load_and_merge``.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional feature columns appended.
    """
    df = df.copy()

    # --- Temperature-efficiency factor --------------------------------------
    # eta_T = 1 + gamma * (T_amb - T_STC)
    # In Nordic climates, cold winters yield eta_T > 1 (efficiency gain).
    df["temp_efficiency_factor"] = (
        1.0 + TEMP_COEFFICIENT_GAMMA * (df["temperature"] - STC_TEMPERATURE)
    )

    # --- Clearness index ----------------------------------------------------
    # k_t = GHI / I_0, where I_0 is extraterrestrial irradiance.
    # I_0 = S * (1 + 0.033 * cos(2 * pi * n / 365))  [Spencer, 1971]
    day_of_year = df.index.dayofyear.astype(float)
    extra_irradiance = SOLAR_CONSTANT * (
        1.0 + 0.033 * np.cos(2.0 * np.pi * day_of_year / 365.0)
    )
    df["extraterrestrial_irradiance"] = extra_irradiance

    with np.errstate(divide="ignore", invalid="ignore"):
        df["clearness_index"] = np.clip(
            df["ghi"] / extra_irradiance, 0.0, 1.0
        )
    df["clearness_index"] = df["clearness_index"].fillna(0.0)

    # --- Cyclical time encodings --------------------------------------------
    hour = df.index.hour + df.index.minute / 60.0
    df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)

    month_float = df.index.month.astype(float)
    df["month_sin"] = np.sin(2.0 * np.pi * month_float / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * month_float / 12.0)

    # --- Drop any rows with NaN that appeared during engineering -------------
    n_before = len(df)
    df = df.dropna()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning("Dropped %d rows with NaN after feature engineering.", n_dropped)

    logger.info("Feature engineering complete. %d features total.", len(df.columns))
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names (everything except target and season)."""
    exclude = {TARGET_COL, "season"}
    return [str(c) for c in df.columns if c not in exclude]
