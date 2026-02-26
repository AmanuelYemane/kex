"""
seasonal_split.py
-----------------
Chronological train/test splitting for the full year and for each
individual Nordic season.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.config import CV_N_SPLITS, SEASONS, TARGET_COL, TRAIN_RATIO

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class SplitData:
    """Container for a single train/test split."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    cv: TimeSeriesSplit


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chronological_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_ratio: float = TRAIN_RATIO,
) -> SplitData:
    """Split a DataFrame chronologically into train and test sets."""
    df = df.sort_index()
    split_idx = int(len(df) * train_ratio)

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    logger.info(
        "Chronological split: %d train rows, %d test rows (ratio=%.2f).",
        len(train_df), len(test_df), train_ratio,
    )

    return SplitData(
        X_train=train_df[feature_cols],
        X_test=test_df[feature_cols],
        y_train=train_df[TARGET_COL],
        y_test=test_df[TARGET_COL],
        cv=TimeSeriesSplit(n_splits=CV_N_SPLITS),
    )


def seasonal_splits(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_ratio: float = TRAIN_RATIO,
) -> dict[str, SplitData]:
    """Create chronological splits for each Nordic season."""
    splits: dict[str, SplitData] = {}

    for season_name, months in SEASONS.items():
        season_df = df[df["season"] == season_name].sort_index()

        if len(season_df) < 20:
            logger.warning(
                "Season '%s' has only %d rows; skipping.",
                season_name, len(season_df),
            )
            continue

        splits[season_name] = chronological_split(
            season_df, feature_cols, train_ratio
        )
        logger.info(
            "Season '%s': %d train, %d test rows.",
            season_name,
            len(splits[season_name].X_train),
            len(splits[season_name].X_test),
        )

    return splits
