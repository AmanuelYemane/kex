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
    """Pass-through for raw features. No engineered features added.

    The DataFrame must have a ``DatetimeIndex`` and at least the columns
    ``temperature`` and ``ghi``.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``data_loader.load_and_merge``.

    Returns
    -------
    pd.DataFrame
        DataFrame with no additional feature columns appended.
    """
    df = df.copy()

    # --- Drop any rows with NaN that appeared during initial loading ---------
    n_before = len(df)
    df = df.dropna()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning("Dropped %d rows with NaN in raw features.", n_dropped)

    logger.info("Raw features locked in. %d features total.", len(df.columns))
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the strictly permitted raw feature columns."""
    return ["temperature", "ghi"]
