"""
models.py
---------
Build, train, and evaluate Ridge (MLR with L2 regularisation) and
Random Forest (RF) pipelines for the comparative analysis.

Ridge is tuned via cross-validation on the training set (RidgeCV).
RF is tuned via GridSearch using the explicit validation set
(PredefinedSplit – so no data from the test period leaks in).
"""

from __future__ import annotations

import logging
from itertools import product
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_SEED, RF_PARAM_GRID, RIDGE_ALPHAS
from src.seasonal_split import SplitData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_mlr() -> Pipeline:
    """Return a Ridge regression pipeline with feature scaling.

    RidgeCV selects the best alpha from RIDGE_ALPHAS using leave-one-out
    cross-validation on the training set (no leakage).
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlr", RidgeCV(alphas=RIDGE_ALPHAS, store_cv_results=False)),
    ])


def build_rf() -> Pipeline:
    """Return a Random Forest pipeline with feature scaling."""
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
) -> tuple[Pipeline, dict[str, Any]]:
    """Fit a pipeline on the training data, optionally with hyperparameter search.

    For **Ridge**: simply fits with RidgeCV (alpha selected internally).

    For **RF** with ``do_grid_search=True``: uses a PredefinedSplit that
    treats the training rows as the "train fold" and the validation rows
    as the "test fold" inside GridSearchCV – no test-set data is ever seen.

    Parameters
    ----------
    pipeline : Pipeline
        An unfitted sklearn Pipeline.
    split : SplitData
        Train / val / test data container.
    param_grid : dict, optional
        Hyperparameter grid for ``GridSearchCV``.
    do_grid_search : bool
        Whether to perform hyperparameter search (RF only).

    Returns
    -------
    tuple[Pipeline, dict[str, Any]]
        The fitted pipeline (or best estimator from grid search) and a 
        dictionary of the best hyperparameters found (if any).
    """
    if do_grid_search and param_grid:
        # Combine train + val into a single array; PredefinedSplit marks
        # train rows with -1 (ignored as test) and val rows with 0 (test fold).
        X_tv = np.vstack([split.X_train.values, split.X_val.values])
        y_tv = np.concatenate([split.y_train.values, split.y_val.values])

        # -1 → held out from the validation fold (used only for fitting)
        #  0 → used as the validation fold
        test_fold = np.concatenate([
            np.full(len(split.X_train), -1, dtype=int),
            np.zeros(len(split.X_val), dtype=int),
        ])
        ps = PredefinedSplit(test_fold)

        logger.info(
            "Running GridSearchCV with PredefinedSplit "
            "(%d train rows, %d val rows) ...",
            len(split.X_train), len(split.X_val),
        )
        search = GridSearchCV(
            pipeline,
            param_grid,
            cv=ps,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=1,
            refit=False,   # we will refit manually on train-only below
        )
        search.fit(X_tv, y_tv)
        best_params = search.best_params_
        logger.info("Best params: %s", best_params)
        logger.info("Best val RMSE: %.4f", -search.best_score_)

        # Refit the best model on training data only (not val)
        pipeline.set_params(**best_params)
        pipeline.fit(split.X_train, split.y_train)
        return pipeline, best_params

    # Default: fit on training set only
    pipeline.fit(split.X_train, split.y_train)
    return pipeline, {}


def predict(pipeline: Pipeline, split: SplitData) -> np.ndarray:
    """Generate predictions on the **test** set."""
    return pipeline.predict(split.X_test)
