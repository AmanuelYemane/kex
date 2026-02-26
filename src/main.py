"""
main.py
-------
End-to-end orchestration for the MLR vs. Random Forest solar power
forecasting comparison across Nordic seasons.

Usage
-----
    # Fetch SMHI data automatically + merge with production CSV
    python3 -m src.main --production data/my_panels.csv

    # Use a specific SMHI station
    python3 -m src.main --production data/my_panels.csv --station 98230

    # Use cached/local SMHI weather CSV (e.g. for testing)
    python3 -m src.main --production data/synthetic_production.csv \\
                        --smhi-csv data/synthetic_smhi.csv

    # With RF hyperparameter tuning
    python3 -m src.main --production data/my_panels.csv --grid-search
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on sys.path when run as ``python src/main.py``
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    FIGURES_DIR,
    RESULTS_DIR,
    RF_PARAM_GRID,
)
from src.data_loader import load_and_merge
from src.evaluation import (
    build_summary_table,
    compute_metrics,
    export_latex,
    residual_analysis,
)
from src.feature_engineering import engineer_features, get_feature_columns
from src.models import build_mlr, build_rf, predict, train_model
from src.seasonal_split import SplitData, chronological_split, seasonal_splits
from src.visualization import (
    plot_actual_vs_predicted,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_qq,
    plot_residuals,
    plot_seasonal_bars,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_model_on_split(
    model_name: str,
    split: SplitData,
    season: str,
    feature_cols: list[str],
    do_grid_search: bool = False,
) -> tuple:
    """Train, predict, evaluate, and plot for a single model + split."""
    if model_name == "MLR":
        pipeline = build_mlr()
        fitted = train_model(pipeline, split)
    else:
        pipeline = build_rf()
        fitted = train_model(
            pipeline,
            split,
            param_grid=RF_PARAM_GRID if do_grid_search else None,
            do_grid_search=do_grid_search,
        )

    y_pred = predict(fitted, split)

    # Metrics (pass X only for MLR to get Breusch-Pagan)
    X_for_bp = split.X_test if model_name == "MLR" else None
    metrics = compute_metrics(split.y_test, y_pred, X=X_for_bp)

    # Residual diagnostics
    diag = residual_analysis(split.y_test, y_pred)

    # Plots
    plot_actual_vs_predicted(
        np.asarray(split.y_test), y_pred, model_name, season
    )
    plot_residuals(y_pred, diag.residuals, model_name, season)
    plot_qq(diag, model_name, season)

    # RF feature importance
    if model_name == "RF":
        rf_step = fitted.named_steps["rf"]
        plot_feature_importance(
            rf_step.feature_importances_, feature_cols, season
        )

    return metrics, fitted


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    production_csv: str,
    smhi_csv: str | None = None,
    do_grid_search: bool = False,
) -> None:
    """Execute the full comparative analysis pipeline.

    Parameters
    ----------
    production_csv : str
        Path to the user's solar production CSV file.
    smhi_csv : str, optional
        Path to a local SMHI weather CSV (bypasses API fetch).
    do_grid_search : bool
        If True, run GridSearchCV for Random Forest hyperparameter tuning.
    """
    # 0. Create output directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and merge production + SMHI data
    print("=" * 60)
    print("STEP 1: Loading production + SMHI weather data")
    print("=" * 60)
    df = load_and_merge(
        production_csv,
        smhi_cache_csv=smhi_csv,
    )
    print(f"  Merged dataset: {len(df)} rows")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")

    # 2. Feature engineering
    print("\n" + "=" * 60)
    print("STEP 2: Engineering features")
    print("=" * 60)
    df = engineer_features(df)
    feature_cols = get_feature_columns(df)
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    # 3. Correlation heatmap (before splitting)
    plot_correlation_heatmap(df, feature_cols)
    print("  Correlation heatmap saved.")

    # 4. Chronological splits
    print("\n" + "=" * 60)
    print("STEP 3: Splitting data (chronological)")
    print("=" * 60)
    full_split = chronological_split(df, feature_cols)
    season_dict = seasonal_splits(df, feature_cols)

    all_splits = {"Full Year": full_split}
    all_splits.update(season_dict)
    print(f"  Splits: {list(all_splits.keys())}")

    # 5. Train and evaluate
    print("\n" + "=" * 60)
    print("STEP 4: Training and evaluating models")
    print("=" * 60)
    results: dict[str, dict[str, object]] = {}

    for split_name, split_data in all_splits.items():
        print(f"\n--- {split_name} ---")
        results[split_name] = {}

        for model_name in ("MLR", "RF"):
            print(f"  Training {model_name} ...")
            metrics, fitted = _run_model_on_split(
                model_name, split_data, split_name, feature_cols,
                do_grid_search=(do_grid_search and model_name == "RF"),
            )
            results[split_name][model_name] = metrics
            print(
                f"    RMSE={metrics.rmse:.4f}  "
                f"nMAE={metrics.nmae:.2f}%  "
                f"R2={metrics.r2:.4f}"
            )

    # 6. Summary table
    print("\n" + "=" * 60)
    print("STEP 5: Exporting results")
    print("=" * 60)
    summary_df = build_summary_table(results)

    csv_path = RESULTS_DIR / "metrics_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"  CSV  -> {csv_path}")

    tex_path = RESULTS_DIR / "metrics_summary.tex"
    export_latex(summary_df, str(tex_path))
    print(f"  LaTeX -> {tex_path}")

    # 7. Seasonal bar charts
    season_only = summary_df[summary_df["season"] != "Full Year"]
    if not season_only.empty:
        plot_seasonal_bars(season_only)
        print("  Seasonal bar charts saved.")

    # 8. Console summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print("\nDone. All figures saved to:", FIGURES_DIR)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="MLR vs. Random Forest: Solar Power Forecasting (Nordic)",
    )
    parser.add_argument(
        "--production", "-p",
        required=True,
        help="Path to the solar production CSV file (your panel data).",
    )
    parser.add_argument(
        "--smhi-csv",
        default=None,
        help="Path to a local SMHI weather CSV (bypasses API fetch).",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        default=False,
        help="Run GridSearchCV for RF hyperparameter tuning (slower).",
    )
    args = parser.parse_args()

    run_pipeline(
        production_csv=args.production,
        smhi_csv=args.smhi_csv,
        do_grid_search=args.grid_search,
    )


if __name__ == "__main__":
    main()
