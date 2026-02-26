"""
models.py
---------
Build, train, and evaluate Multiple Linear Regression (MLR) and
Random Forest (RF) pipelines for the comparative analysis.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_SEED, RF_PARAM_GRID
from src.seasonal_split import SplitData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_mlr() -> Pipeline:
    """Return a Multiple Linear Regression pipeline with feature scaling."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlr", LinearRegression()),
    ])


def build_rf(param_grid: dict[str, list] | None = None) -> Pipeline:
    """Return a Random Forest pipeline with feature scaling."""
    if param_grid is None:
        param_grid = RF_PARAM_GRID

    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1)),
    ])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    pipeline: Pipeline,
    split: SplitData,
    *,
    param_grid: dict[str, list] | None = None,
    do_grid_search: bool = False,
) -> Pipeline:
    """Fit a pipeline on the training data, optionally with grid search.

    Parameters
    ----------
    pipeline : Pipeline
        An unfitted sklearn Pipeline.
    split : SplitData
        Train/test data container.
    param_grid : dict, optional
        Grid for ``GridSearchCV``.
    do_grid_search : bool
        Whether to perform ``GridSearchCV`` with ``TimeSeriesSplit``.

    Returns
    -------
    Pipeline
        The fitted pipeline (or the best estimator from grid search).
    """
    if do_grid_search and param_grid:
        logger.info("Running GridSearchCV with %d folds ...", split.cv.n_splits)
        search = GridSearchCV(
            pipeline,
            param_grid,
            cv=split.cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=1,
        )
        search.fit(split.X_train, split.y_train)
        logger.info("Best params: %s", search.best_params_)
        logger.info("Best CV RMSE: %.4f", -search.best_score_)
        return search.best_estimator_

    pipeline.fit(split.X_train, split.y_train)
    return pipeline


def predict(pipeline: Pipeline, split: SplitData) -> np.ndarray:
    """Generate predictions on the test set."""
    return pipeline.predict(split.X_test)
