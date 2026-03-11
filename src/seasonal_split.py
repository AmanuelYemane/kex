"""
seasonal_split.py
-----------------
Explicit train / validation / test splitting for the full year and for each
individual Nordic season.

Split boundaries are taken from ``config.py``:
  - Train:      TRAIN_START – TRAIN_END   (2017-03-01 – 2022-02-28)
  - Validation: VAL_START   – VAL_END     (2022-03-01 – 2023-02-28)
  - Test:       TEST_START  – TEST_END    (2023-03-01 – 2024-02-28)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src.config import (
    SEASONS,
    TARGET_COL,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _date_slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Return rows of *df* (timezone-aware DatetimeIndex) between [start, end]."""
    tz = df.index.tz
    start_ts = pd.Timestamp(start, tz=tz)
    end_ts   = pd.Timestamp(end,   tz=tz) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return df.loc[start_ts:end_ts]


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class SplitData:
    """Container for train / validation / test splits."""

    # Training set
    X_train: pd.DataFrame
    y_train: pd.Series

    # Validation set (used for RF hyperparameter tuning)
    X_val: pd.DataFrame
    y_val: pd.Series

    # Test set (used for final evaluation)
    X_test: pd.DataFrame
    y_test: pd.Series


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chronological_split(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> SplitData:
    """Split a DataFrame into train / val / test using the configured date boundaries.

    Parameters
    ----------
    df : pd.DataFrame
        Full merged dataset with a timezone-aware DatetimeIndex.
    feature_cols : list[str]
        Column names to use as model features.

    Returns
    -------
    SplitData
    """
    df = df.sort_index()

    train_df = _date_slice(df, TRAIN_START, TRAIN_END)
    val_df   = _date_slice(df, VAL_START,   VAL_END)
    test_df  = _date_slice(df, TEST_START,  TEST_END)

    logger.info(
        "Chronological split: %d train, %d val, %d test rows.",
        len(train_df), len(val_df), len(test_df),
    )

    return SplitData(
        X_train=train_df[feature_cols],
        y_train=train_df[TARGET_COL],
        X_val=val_df[feature_cols],
        y_val=val_df[TARGET_COL],
        X_test=test_df[feature_cols],
        y_test=test_df[TARGET_COL],
    )


def seasonal_splits(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, SplitData]:
    """Create train/val/test splits for each Nordic season.

    Each season's data is filtered by month, *then* split by the global
    date boundaries so the chronological ordering is preserved.

    Parameters
    ----------
    df : pd.DataFrame
        Full merged dataset with a timezone-aware DatetimeIndex.
    feature_cols : list[str]
        Column names to use as model features.

    Returns
    -------
    dict[str, SplitData]
        Keys are season names (e.g. ``"Winter"``).
    """
    splits: dict[str, SplitData] = {}

    for season_name in SEASONS:
        season_df = df[df["season"] == season_name].sort_index()

        train_df = _date_slice(season_df, TRAIN_START, TRAIN_END)
        val_df   = _date_slice(season_df, VAL_START,   VAL_END)
        test_df  = _date_slice(season_df, TEST_START,  TEST_END)

        # Skip if any partition is too small to be meaningful
        min_rows = min(len(train_df), len(val_df), len(test_df))
        if min_rows < 10:
            logger.warning(
                "Season '%s' has a partition with only %d rows; skipping.",
                season_name, min_rows,
            )
            continue

        splits[season_name] = SplitData(
            X_train=train_df[feature_cols],
            y_train=train_df[TARGET_COL],
            X_val=val_df[feature_cols],
            y_val=val_df[TARGET_COL],
            X_test=test_df[feature_cols],
            y_test=test_df[TARGET_COL],
        )
        logger.info(
            "Season '%s': %d train, %d val, %d test rows.",
            season_name, len(train_df), len(val_df), len(test_df),
        )

    return splits
