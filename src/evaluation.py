"""
evaluation.py
-------------
Compute regression metrics and residual diagnostics for the comparative
analysis between RR and Random Forest.

Metrics
-------
- nRMSE: Normalised Root Mean Squared Error (RMSE / mean(y) * 100 %)
- nMAE: Normalised Mean Absolute Error (MAE / mean(y) * 100 %)
- R^2: Coefficient of determination
- Breusch-Pagan p-value: heteroscedasticity test on RR residuals
- Durbin-Watson: autocorrelation in residuals
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    """Container for a single model's evaluation metrics."""

    nrmse: float  # percentage
    nmae: float  # percentage
    r2: float
    breusch_pagan_pvalue: float | None = None
    durbin_watson_stat: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "nrmse_pct": self.nrmse,
            "nmae_pct": self.nmae,
            "r2": self.r2,
        }


@dataclass
class ResidualDiagnostics:
    """Container for residual analysis outputs."""

    residuals: np.ndarray
    qq_theoretical: np.ndarray
    qq_sample: np.ndarray
    durbin_watson_stat: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: np.ndarray | pd.DataFrame | None = None,
) -> Metrics:
    """Compute nRMSE, nMAE, R^2, and (optionally) heteroscedasticity stats.

    Parameters
    ----------
    y_true : array-like
        Ground-truth target values.
    y_pred : array-like
        Model predictions.
    X : array-like, optional
        Feature matrix corresponding to ``y_true``.  Required for the
        Breusch-Pagan test (tests whether residual variance depends on X).

    Returns
    -------
    Metrics
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mean_y = float(np.mean(y_true))
    nrmse = (rmse / mean_y * 100.0) if mean_y != 0 else float("inf")
    nmae = (mae / mean_y * 100.0) if mean_y != 0 else float("inf")
    r2 = float(r2_score(y_true, y_pred))

    bp_pvalue = None
    dw_stat = None

    if X is not None:
        residuals = y_true - y_pred
        try:
            import statsmodels.api as sm
            exog_with_const = sm.add_constant(np.asarray(X))
            _, bp_pvalue, _, _ = het_breuschpagan(residuals, exog_with_const)
            bp_pvalue = float(bp_pvalue)
        except Exception as exc:
            logger.warning("Breusch-Pagan test failed: %s", exc)

        dw_stat = float(durbin_watson(residuals))

    logger.info(
        "Metrics: nRMSE=%.2f%%, nMAE=%.2f%%, R2=%.4f", nrmse, nmae, r2
    )
    return Metrics(
        nrmse=nrmse,
        nmae=nmae,
        r2=r2,
        breusch_pagan_pvalue=bp_pvalue,
        durbin_watson_stat=dw_stat,
    )


def residual_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> ResidualDiagnostics:
    """Compute residual diagnostics for normality and autocorrelation.

    Returns
    -------
    ResidualDiagnostics
        Contains raw residuals, Q-Q coordinates, and Durbin-Watson statistic.
    """
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    (qq_theoretical, qq_sample), _ = sp_stats.probplot(residuals, dist="norm")

    return ResidualDiagnostics(
        residuals=residuals,
        qq_theoretical=qq_theoretical,
        qq_sample=qq_sample,
        durbin_watson_stat=float(durbin_watson(residuals)),
    )


def build_summary_table(
    results: dict[str, dict[str, Metrics]],
) -> pd.DataFrame:
    """Build a tidy DataFrame from nested results.

    Parameters
    ----------
    results : dict
        ``{season: {model_name: Metrics, ...}, ...}``

    Returns
    -------
    pd.DataFrame
        Columns: season, model, nrmse_pct, nmae_pct, r2, bp_pvalue, dw_stat.
    """
    rows: list[dict[str, Any]] = []
    for season, models in results.items():
        for model_name, metrics in models.items():
            row = {"season": season, "model": model_name}
            row.update(metrics.to_dict())
            rows.append(row)

    return pd.DataFrame(rows)


def export_latex(df: pd.DataFrame, filepath: str) -> None:
    """Export a metrics DataFrame as a LaTeX table.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``build_summary_table``.
    filepath : str
        Destination ``.tex`` file.
    """
    formatters = {
        "nrmse_pct": "{:.2f}".format,
        "nmae_pct": "{:.2f}".format,
        "r2": "{:.4f}".format,
        "bp_pvalue": lambda v: f"{v:.4f}" if pd.notna(v) else "--",
        "dw_stat": lambda v: f"{v:.4f}" if pd.notna(v) else "--",
    }

    latex_str = df.to_latex(
        index=False,
        formatters=formatters,
        caption=(
            "Comparative regression metrics for RR and RF across Nordic seasons."
        ),
        label="tab:metrics_summary",
        column_format="llccccc",
        escape=False,
    )

    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(latex_str)

    logger.info("LaTeX table written to %s", filepath)
