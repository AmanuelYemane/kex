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
    SMHI_STATION_GHI,
    SMHI_STATION_TEMP,
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
    """Parse the SMHI JSON response (``latest-months``) into a DataFrame.

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


def _parse_smhi_csv(text: str, parameter: int) -> pd.DataFrame:
    """Parse the SMHI semicolon-delimited CSV (``corrected-archive``).

    The CSV has a multi-line header (station info, parameter description,
    column names) followed by data rows.  Data lines look like:

        Datum;Tid (UTC);Value;Kvalitet;;extra info...

    We detect the header row starting with ``Datum;`` and parse from there.
    """
    import io
    import re

    lines = text.splitlines()

    # Find the header line that starts with "Datum;"
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Datum;"):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(
            f"Could not find data header in corrected-archive CSV "
            f"for parameter {parameter}."
        )

    # Data rows follow the header.  Each row is:
    # date ; time(UTC) ; value ; quality ; ; optional_annotation
    # Parse only lines that start with a date pattern (YYYY-MM-DD)
    date_re = re.compile(r"^\d{4}-\d{2}-\d{2};")
    records = []
    for line in lines[header_idx + 1 :]:
        if not date_re.match(line):
            continue
        parts = line.split(";")
        if len(parts) < 4:
            continue
        date_str = parts[0]       # e.g. "2024-06-15"
        time_str = parts[1]       # e.g. "14:00:00"
        value_str = parts[2]      # e.g. "18.3"
        quality = parts[3]        # e.g. "G" or "Y"

        if quality not in ("G", "Y"):
            continue
        try:
            val = float(value_str)
        except ValueError:
            continue

        ts = pd.Timestamp(f"{date_str} {time_str}", tz="UTC")
        records.append({"timestamp": ts, "value": val})

    if not records:
        raise ValueError(
            f"No valid data rows after parsing corrected-archive CSV "
            f"for parameter {parameter}."
        )

    df = pd.DataFrame(records).set_index("timestamp").sort_index()
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Periods to try in order: corrected-archive (CSV, years of data) first,
# then latest-months (JSON, ~4 months of recent data) to fill the gap.
_PERIODS = ["corrected-archive", "latest-months"]


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

    # Fetch from API — try each period and combine results.
    # corrected-archive → data.csv  (semicolon-delimited)
    # latest-months     → data.json (JSON)
    all_frames = []
    for period in _PERIODS:
        if period == "corrected-archive":
            ext = "csv"
        else:
            ext = "json"

        url = (
            f"{SMHI_BASE_URL}/parameter/{parameter}"
            f"/station/{station_id}/period/{period}/data.{ext}"
        )
        logger.info("Fetching SMHI data: %s", url)
        resp = requests.get(url, timeout=120)
        if resp.status_code == 404:
            logger.warning(
                "Period '%s' not available for station=%d param=%d, trying next.",
                period, station_id, parameter,
            )
            continue
        resp.raise_for_status()

        if ext == "json":
            all_frames.append(_parse_smhi_json(resp.json(), parameter))
        else:
            all_frames.append(_parse_smhi_csv(resp.text, parameter))

    if not all_frames:
        raise RuntimeError(
            f"No data available for station {station_id}, parameter {parameter}. "
            f"Tried periods: {_PERIODS}. Check that the station ID is valid for "
            f"this parameter at https://opendata-download-metobs.smhi.se"
        )

    # Combine and deduplicate (corrected-archive + latest-months may overlap)
    df = pd.concat(all_frames)
    df = df[~df.index.duplicated(keep="first")].sort_index()

    # Save to cache
    df.to_csv(cache)
    logger.info(
        "Cached %d rows to %s", len(df), cache
    )
    return df


def fetch_weather_data(
    temp_station_id: int | None = None,
    ghi_station_id: int | None = None,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch temperature and GHI from SMHI, merge into one DataFrame.

    SMHI uses separate station IDs for temperature and GHI.  If not
    supplied, the defaults from ``config`` are used.

    Parameters
    ----------
    temp_station_id : int, optional
        Station ID for temperature data (default: ``config.SMHI_STATION_TEMP``).
    ghi_station_id : int, optional
        Station ID for GHI data (default: ``config.SMHI_STATION_GHI``).
    use_cache : bool
        Whether to use local cache.

    Returns
    -------
    pd.DataFrame
        Columns: ``temperature`` (°C), ``ghi`` (W/m²).
        Index: ``timestamp`` (UTC).
    """
    if temp_station_id is None:
        temp_station_id = SMHI_STATION_TEMP
    if ghi_station_id is None:
        ghi_station_id = SMHI_STATION_GHI

    temp_df = fetch_smhi_parameter(
        temp_station_id, SMHI_PARAM_TEMPERATURE, use_cache=use_cache
    )
    temp_df = temp_df.rename(columns={"value": "temperature"})

    ghi_df = fetch_smhi_parameter(
        ghi_station_id, SMHI_PARAM_GHI, use_cache=use_cache
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

