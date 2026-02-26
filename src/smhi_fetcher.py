"""
smhi_fetcher.py
---------------
Fetch meteorological observation data from the SMHI Open Data API and cache
it locally to avoid repeated network calls.

SMHI API structure (Meteorological Observations v1.0)
-----------------------------------------------------
Base: https://opendata-download-metobs.smhi.se/api/version/1.0

Endpoints used:
  - /parameter/{param}/station-set/all.json          -> list all stations
  - /parameter/{param}/station/{id}/period/corrected-archive/data.json
                                                      -> historical data
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from src.config import (
    SMHI_BASE_URL,
    SMHI_CACHE_DIR,
    SMHI_PARAM_GHI,
    SMHI_PARAM_TEMPERATURE,
    SMHI_STATION_ID,
)

logger = logging.getLogger(__name__)

# Cache validity: 24 hours (seconds)
_CACHE_TTL = 86400


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cache_path(station_id: int, parameter: int) -> Path:
    """Return the cache file path for a given station + parameter."""
    SMHI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return SMHI_CACHE_DIR / f"station_{station_id}_param_{parameter}.csv"


def _cache_is_fresh(path: Path) -> bool:
    """Check whether a cached file exists and is less than 24 h old."""
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < _CACHE_TTL


def _parse_smhi_json(data: dict, parameter: int) -> pd.DataFrame:
    """Parse the SMHI JSON response into a tidy DataFrame.

    The response ``data`` dict contains a ``value`` list where each entry has:
    - ``date``: Unix timestamp in milliseconds
    - ``value``: the measured value as a string
    - ``quality``: quality code ("G" = approved, "Y" = suspect, etc.)
    """
    values = data.get("value", [])
    if not values:
        raise ValueError(
            f"No data points returned for parameter {parameter}."
        )

    records = []
    for entry in values:
        ts_ms = entry["date"]
        val = entry["value"]
        quality = entry.get("quality", "")
        records.append({
            "timestamp": pd.Timestamp(ts_ms, unit="ms", tz="UTC"),
            "value": float(val),
            "quality": quality,
        })

    df = pd.DataFrame(records)
    # Keep only approved or controlled data (quality codes G, Y)
    df = df[df["quality"].isin(["G", "Y"])].drop(columns=["quality"])
    df = df.set_index("timestamp").sort_index()
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_smhi_parameter(
    station_id: int,
    parameter: int,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch a single SMHI parameter for a station.

    Parameters
    ----------
    station_id : int
        SMHI station numeric ID (e.g. 98230 for Stockholm).
    parameter : int
        SMHI parameter number (1 = temperature, 11 = GHI).
    use_cache : bool
        If True, use locally cached CSV if available and fresh.

    Returns
    -------
    pd.DataFrame
        Index: ``timestamp`` (UTC), single column ``value``.
    """
    cache = _cache_path(station_id, parameter)

    # Try cache first
    if use_cache and _cache_is_fresh(cache):
        logger.info(
            "Loading cached SMHI data: station=%d, param=%d", station_id, parameter
        )
        df = pd.read_csv(cache, parse_dates=["timestamp"], index_col="timestamp")
        return df

    # Fetch from API
    url = (
        f"{SMHI_BASE_URL}/parameter/{parameter}"
        f"/station/{station_id}/period/corrected-archive/data.json"
    )
    logger.info("Fetching SMHI data: %s", url)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    df = _parse_smhi_json(resp.json(), parameter)

    # Save to cache
    df.to_csv(cache)
    logger.info(
        "Cached %d rows to %s", len(df), cache
    )
    return df


def fetch_weather_data(
    station_id: int | None = None,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch temperature and GHI from SMHI, merge into one DataFrame.

    Parameters
    ----------
    station_id : int, optional
        SMHI station ID. Defaults to ``config.SMHI_STATION_ID``.
    use_cache : bool
        Whether to use local cache.

    Returns
    -------
    pd.DataFrame
        Columns: ``temperature`` (°C), ``ghi`` (W/m²).
        Index: ``timestamp`` (UTC).
    """
    if station_id is None:
        station_id = SMHI_STATION_ID

    temp_df = fetch_smhi_parameter(
        station_id, SMHI_PARAM_TEMPERATURE, use_cache=use_cache
    )
    temp_df = temp_df.rename(columns={"value": "temperature"})

    ghi_df = fetch_smhi_parameter(
        station_id, SMHI_PARAM_GHI, use_cache=use_cache
    )
    ghi_df = ghi_df.rename(columns={"value": "ghi"})

    # Inner-join on timestamp: keep only hours where both are available
    merged = temp_df.join(ghi_df, how="inner")

    logger.info(
        "SMHI weather data: %d matched rows (temp=%d, ghi=%d), range %s to %s.",
        len(merged), len(temp_df), len(ghi_df),
        merged.index.min(), merged.index.max(),
    )
    return merged


def list_stations(parameter: int = SMHI_PARAM_TEMPERATURE) -> pd.DataFrame:
    """List all SMHI stations that have data for a given parameter.

    Returns
    -------
    pd.DataFrame
        Columns: ``station_id``, ``name``, ``latitude``, ``longitude``,
        ``active``.
    """
    url = f"{SMHI_BASE_URL}/parameter/{parameter}.json"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    stations = resp.json().get("station", [])
    rows = []
    for s in stations:
        rows.append({
            "station_id": int(s["key"]),
            "name": s.get("name", ""),
            "latitude": s.get("latitude"),
            "longitude": s.get("longitude"),
            "active": s.get("active", False),
        })

    return pd.DataFrame(rows)
