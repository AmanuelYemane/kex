"""
data_loader.py
--------------
Load the user's solar production CSV, fetch SMHI weather data, and merge
them into a single pipeline-ready DataFrame.

The two data sources are:
1. **Production data**: user-provided CSV with at least a timestamp column
   and a power output column (names configurable in ``config.py``).
2. **SMHI weather data**: temperature + GHI fetched via ``smhi_fetcher``.

Data is filtered to the range [DATA_START, DATA_END] defined in config.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DATA_END,
    DATA_START,
    MONTH_TO_SEASON,
    PRODUCTION_POWER_COL,
    PRODUCTION_TIMESTAMP_COL,
    RAW_FEATURE_COLS,
    TARGET_COL,
)
from src.smhi_fetcher import fetch_weather_data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_production_data(filepath: str | Path) -> pd.DataFrame:
    """Load and clean the user's solar production CSV.

    The real production CSV (from Stockholm Stad) uses:
    - UTF-8-BOM encoding
    - Semicolon separator
    - Column headers: ``Tid`` (timestamp), ``energiprod_sum`` (kWh per hour)
    - Hourly energy in kWh ≡ average power in kW (1 kWh over 1 h = 1 kW avg)

    Parameters
    ----------
    filepath : str or Path
        Path to the production CSV file.

    Returns
    -------
    pd.DataFrame
        Indexed by datetime (UTC), single column ``power_kw``.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Production data file not found: {filepath}")

    logger.info("Loading production data from %s", filepath)

    # Try reading with the real format first (semicolon, BOM).
    # Fall back to comma-separated if the expected timestamp column is missing.
    df = pd.read_csv(
        filepath,
        sep=";",
        encoding="utf-8-sig",  # strips BOM automatically
        parse_dates=[PRODUCTION_TIMESTAMP_COL],
    )

    # Rename to internal standard names
    df = df.rename(columns={
        PRODUCTION_TIMESTAMP_COL: "timestamp",
        PRODUCTION_POWER_COL: TARGET_COL,
    })

    # Drop columns that are not needed (trailing empties, embedded weather cols)
    df = df[["timestamp", TARGET_COL]]

    df = df.set_index("timestamp").sort_index()

    # Coerce numeric (some rows may have empty strings after semicolons)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    # Ensure timezone-aware (assume UTC if naive)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    # Remove duplicates
    n_dup = df.index.duplicated().sum()
    if n_dup > 0:
        logger.warning("Dropping %d duplicate timestamps in production data.", n_dup)
        df = df[~df.index.duplicated(keep="first")]

    logger.info(
        "Production data: %d rows, range %s to %s.",
        len(df), df.index.min(), df.index.max(),
    )
    return df


def load_and_merge(
    production_csv: str | Path,
    temp_station_id: int | None = None,
    ghi_station_id: int | None = None,
    *,
    use_cache: bool = True,
    smhi_cache_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Load production data and SMHI weather data, merge on timestamp.

    Parameters
    ----------
    production_csv : str or Path
        Path to the user's solar production CSV.
    temp_station_id : int, optional
        SMHI station ID for temperature. Defaults to ``config.SMHI_STATION_TEMP``.
    ghi_station_id : int, optional
        SMHI station ID for GHI. Defaults to ``config.SMHI_STATION_GHI``.
    use_cache : bool
        Whether to use cached SMHI data.
    smhi_cache_csv : str or Path, optional
        If provided, load SMHI weather data from this CSV instead of
        fetching from the API (used for testing with synthetic data).

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with columns: ``temperature``, ``ghi``,
        ``power_kw``, ``season``.  Indexed by datetime (UTC).
        Rows are strictly within [DATA_START, DATA_END].
    """
    # Load production data
    prod_df = load_production_data(production_csv)

    # Load SMHI weather data
    if smhi_cache_csv is not None:
        logger.info("Loading SMHI data from local file: %s", smhi_cache_csv)
        weather_df = pd.read_csv(
            smhi_cache_csv,
            parse_dates=["timestamp"],
            index_col="timestamp",
        )
        if weather_df.index.tz is None:
            weather_df.index = weather_df.index.tz_localize("UTC")
    else:
        weather_df = fetch_weather_data(
            temp_station_id, ghi_station_id, use_cache=use_cache
        )

    # Merge on nearest hourly timestamp (tolerance: 30 minutes)
    prod_df = prod_df.sort_index()
    weather_df = weather_df.sort_index()

    merged = pd.merge_asof(
        prod_df,
        weather_df,
        left_index=True,
        right_index=True,
        tolerance=pd.Timedelta("30min"),
        direction="nearest",
    )

    n_before = len(merged)
    merged = merged.dropna(subset=RAW_FEATURE_COLS)
    n_dropped = n_before - len(merged)
    if n_dropped > 0:
        logger.warning(
            "Dropped %d rows with no matching SMHI data (within 30 min).",
            n_dropped,
        )

    # Forward-fill small gaps (<=3 consecutive NaN) in the target
    merged[TARGET_COL] = merged[TARGET_COL].ffill(limit=3)
    merged = merged.dropna(subset=[TARGET_COL])

    # Remove nighttime rows (no irradiance and no power)
    nighttime_mask = (merged["ghi"] <= 0) & (merged[TARGET_COL] <= 0)
    n_night = nighttime_mask.sum()
    if n_night > 0:
        logger.info("Removing %d nighttime rows.", n_night)
        merged = merged[~nighttime_mask]

    # -----------------------------------------------------------------------
    # Filter strictly to the configured data range [DATA_START, DATA_END].
    # Both bounds are inclusive (date-level slicing via .loc on a
    # timezone-aware index requires timezone-aware boundary strings).
    # -----------------------------------------------------------------------
    start_ts = pd.Timestamp(DATA_START, tz="UTC")
    end_ts   = pd.Timestamp(DATA_END,   tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    merged = merged.loc[start_ts:end_ts]

    n_filtered = len(merged)
    logger.info(
        "After date filter [%s, %s]: %d rows remain.",
        DATA_START, DATA_END, n_filtered,
    )

    # Add season column
    merged["season"] = merged.index.month.map(MONTH_TO_SEASON)

    logger.info(
        "Merged dataset: %d rows, date range %s to %s.",
        len(merged), merged.index.min(), merged.index.max(),
    )
    return merged
