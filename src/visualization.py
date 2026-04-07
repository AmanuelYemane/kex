"""
visualization.py
----------------
Publication-quality figures for the RR vs. Random Forest solar power
forecasting comparison.  All plots are saved to ``config.FIGURES_DIR``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURE_DPI, FIGURE_FORMAT, FIGURES_DIR, PLOT_STYLE
from src.evaluation import Metrics, ResidualDiagnostics

logger = logging.getLogger(__name__)

plt.style.use(PLOT_STYLE)

MODEL_COLOURS = {"RR": "#3498db", "RF": "#e74c3c"}
SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]


def _ensure_dir() -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


# ---------------------------------------------------------------------------
# 1. Actual vs Predicted Scatter
# ---------------------------------------------------------------------------

def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    season: str,
    *,
    ax: plt.Axes | None = None,
    save: bool = True,
) -> plt.Figure | None:
    """Scatter plot of actual vs. predicted with a 45-degree reference line."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(y_true, y_pred, alpha=0.35, s=12, edgecolors="none",
               color=MODEL_COLOURS.get(model_name, "#555"))
    lims = [
        min(np.min(y_true), np.min(y_pred)),
        max(np.max(y_true), np.max(y_pred)),
    ]
    ax.plot(lims, lims, "--", color="#2c3e50", linewidth=1.0, label="Ideal")
    ax.set_xlabel("Actual Power (kW)")
    ax.set_ylabel("Predicted Power (kW)")
    ax.set_title(f"{model_name}: Actual vs. Predicted ({season})")
    ax.legend(loc="upper left")
    ax.set_aspect("equal", adjustable="box")

    if save and fig is not None:
        outpath = _ensure_dir() / f"scatter_{model_name}_{season}.{FIGURE_FORMAT}"
        fig.tight_layout()
        fig.savefig(outpath, dpi=FIGURE_DPI)
        plt.close(fig)
        logger.info("Saved %s", outpath)

    return fig


# ---------------------------------------------------------------------------
# 2. Residual Plot
# ---------------------------------------------------------------------------

def plot_residuals(
    y_pred: np.ndarray,
    residuals: np.ndarray,
    model_name: str,
    season: str,
) -> None:
    """Residuals vs. predicted values with a horizontal zero-line."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, residuals, alpha=0.35, s=12, edgecolors="none",
               color=MODEL_COLOURS.get(model_name, "#555"))
    ax.axhline(0, color="#2c3e50", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Predicted Power (kW)")
    ax.set_ylabel("Residual (kW)")
    ax.set_title(f"{model_name}: Residuals ({season})")
    fig.tight_layout()

    outpath = _ensure_dir() / f"residuals_{model_name}_{season}.{FIGURE_FORMAT}"
    fig.savefig(outpath, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved %s", outpath)


# ---------------------------------------------------------------------------
# 3. Seasonal Bar Charts (nRMSE, nMAE and R^2)
# ---------------------------------------------------------------------------

def plot_seasonal_bars(summary_df: pd.DataFrame) -> None:
    """Bar charts of nRMSE, nMAE, and R^2 grouped by season (test set), split into separate plots."""
    # Ensure Full Year is displayed alongside the seasons
    present_categories = [s for s in ["Full Year"] + SEASON_ORDER if s in summary_df["season"].values]
    plot_df = summary_df[summary_df["season"].isin(present_categories)].copy()
    plot_df["season"] = pd.Categorical(
        plot_df["season"], categories=present_categories, ordered=True
    )

    metrics = [
        ("nrmse_pct", "nRMSE (%)", "nRMSE Performance Comparison"),
        ("nmae_pct", "nMAE (%)", "nMAE Performance Comparison"),
        ("r2", r"$R^2$", r"$R^2$ Performance Comparison")
    ]

    for col, ylabel, title in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=plot_df, x="season", y=col, hue="model",
                    palette=MODEL_COLOURS, ax=ax)
        ax.set_title(title, fontsize=14)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        
        # Place legend nicely
        ax.legend(title="Model", loc="best")
        
        fig.tight_layout()
        outpath = _ensure_dir() / f"bar_{col}.{FIGURE_FORMAT}"
        fig.savefig(outpath, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", outpath)


# ---------------------------------------------------------------------------
# 4. RF Feature Importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    season: str,
    top_n: int = 15,
) -> None:
    """Horizontal bar chart of Random Forest feature importances."""
    indices = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(len(indices)), importances[indices],
            color=MODEL_COLOURS["RF"], edgecolor="white")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Feature Importance (MDI)")
    ax.set_title(f"RF Feature Importance ({season})")
    fig.tight_layout()

    outpath = _ensure_dir() / f"feature_importance_RF_{season}.{FIGURE_FORMAT}"
    fig.savefig(outpath, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved %s", outpath)


# ---------------------------------------------------------------------------
# 5. Q-Q Plot
# ---------------------------------------------------------------------------

def plot_qq(
    diag: ResidualDiagnostics,
    model_name: str,
    season: str,
) -> None:
    """Q-Q plot of residuals against a normal distribution."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(diag.qq_theoretical, diag.qq_sample, s=12, alpha=0.6,
               color=MODEL_COLOURS.get(model_name, "#555"), edgecolors="none")
    mn, mx = np.min(diag.qq_theoretical), np.max(diag.qq_theoretical)
    ax.plot([mn, mx], [mn, mx], "--", color="#2c3e50", linewidth=1.0)
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.set_title(f"{model_name}: Q-Q Plot ({season})")
    fig.tight_layout()

    outpath = _ensure_dir() / f"qq_{model_name}_{season}.{FIGURE_FORMAT}"
    fig.savefig(outpath, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved %s", outpath)


# ---------------------------------------------------------------------------
# 6. Correlation Heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    """Heatmap of Pearson correlations among engineered features."""
    corr = df[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()

    outpath = _ensure_dir() / f"correlation_heatmap.{FIGURE_FORMAT}"
    fig.savefig(outpath, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved %s", outpath)
