"""
Standalone script to fetch and save real SMHI weather data.

Usage:
    python3 -m src.fetch_smhi
    python3 -m src.fetch_smhi --station 98735
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, SMHI_STATION_ID
from src.smhi_fetcher import fetch_weather_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch SMHI weather data")
    parser.add_argument(
        "--station", "-s", type=int, default=SMHI_STATION_ID,
        help=f"SMHI station ID (default: {SMHI_STATION_ID})",
    )
    args = parser.parse_args()

    print(f"Fetching temperature + GHI from SMHI station {args.station} ...")
    df = fetch_weather_data(args.station, use_cache=False)

    outpath = DATA_DIR / f"smhi_station_{args.station}.csv"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(outpath)

    print(f"\nSaved {len(df)} rows to {outpath}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nLast 5 rows:")
    print(df.tail())


if __name__ == "__main__":
    main()
