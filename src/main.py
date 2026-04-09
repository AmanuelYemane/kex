"""
main.py
-------
End-to-end orchestration for the Ridge vs. Random Forest solar power
forecasting comparison across Nordic seasons.

Usage
-----
    # Fetch SMHI data automatically + merge with production CSV
    python3 -m src.main --production data/production.csv

    # Use cached/local SMHI weather CSV
    python3 -m src.main --production data/production.csv \\
                        --smhi-csv data/smhi_temp98230_ghi98735.csv

    # With RF hyperparameter tuning (slower)
    python3 -m src.main --production data/production.csv --grid-search
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

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
    SMHI_STATION_GHI,
    SMHI_STATION_TEMP,
)
from src.data_loader import load_and_merge
from src.evaluation import (
    build_summary_table,
    compute_metrics,
    export_latex,
    residual_analysis,
)
from src.feature_engineering import engineer_features, get_feature_columns
from src.models import build_rr, build_rf, predict, train_model
from src.seasonal_split import SplitData, chronological_split, seasonal_splits
from src.visualization import (
    plot_correlation_heatmap,
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
    preloaded_params: dict[str, Any] | None = None,
):
    """Train, tune (if RF), predict on test set, evaluate and plot.

    - Ridge: trained on split.X_train / y_train; evaluated on X_test / y_test.
    - RF:    tuned using the val set (when do_grid_search=True), then refitted
             on train-only; evaluated on X_test / y_test.
    """
    if model_name == "RR":
        pipeline = build_rr()
        fitted, best_params = train_model(pipeline, split)
    else:
        pipeline = build_rf()
        if preloaded_params:
            pipeline.set_params(**preloaded_params)
        fitted, best_params = train_model(
            pipeline,
            split,
            param_grid=RF_PARAM_GRID if do_grid_search else None,
            do_grid_search=do_grid_search,
        )

    # Predict on TEST set only
    y_pred = predict(fitted, split)
    if model_name == "RR":
        y_pred = np.maximum(y_pred, 0.0)

    # Metrics on test set (pass X_test for Breusch-Pagan on Ridge/RR)
    X_for_bp = split.X_test if model_name == "RR" else None
    metrics = compute_metrics(split.y_test, y_pred, X=X_for_bp)

    # Residual diagnostics on test set
    diag = residual_analysis(split.y_test, y_pred)

    # Plots (all based on test-set predictions)
    plot_residuals(y_pred, diag.residuals, model_name, season)
    plot_qq(diag, model_name, season)

    # RF feature importance
    rf_importance = None
    if model_name == "RF":
        rf_step = fitted.named_steps["rf"]
        rf_importance = rf_step.feature_importances_

    return metrics, fitted, best_params, rf_importance, np.asarray(split.y_test), y_pred


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    production_csv: str,
    temp_station_id: int = SMHI_STATION_TEMP,
    ghi_station_id: int = SMHI_STATION_GHI,
    smhi_csv: str | None = None,
    do_grid_search: bool = False,
) -> None:
    """Execute the full comparative analysis pipeline.

    Parameters
    ----------
    production_csv : str
        Path to the user's solar production CSV file.
    temp_station_id : int
        SMHI station ID for temperature data.
    ghi_station_id : int
        SMHI station ID for GHI data.
    smhi_csv : str, optional
        Path to a local SMHI weather CSV (bypasses API fetch).
    do_grid_search : bool
        If True, run grid search for Random Forest hyperparameter tuning
        using the explicit validation set.
    """
    # 0. Create output directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and merge production + SMHI data (filtered to configured range)
    print("=" * 60)
    print("STEP 1: Loading production + SMHI weather data")
    print("=" * 60)
    df = load_and_merge(
        production_csv,
        temp_station_id=temp_station_id,
        ghi_station_id=ghi_station_id,
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

    # 3. Correlation heatmap (computed on full dataset before splitting)
    plot_correlation_heatmap(df, feature_cols)
    print("  Correlation heatmap saved.")

    # 4. Explicit train / val / test splits (date-based)
    print("\n" + "=" * 60)
    print("STEP 3: Splitting data (train / val / test)")
    print("=" * 60)
    full_split = chronological_split(df, feature_cols)
    season_dict = seasonal_splits(df, feature_cols)

    print(
        f"  Full year — train: {len(full_split.X_train)}, "
        f"val: {len(full_split.X_val)}, "
        f"test: {len(full_split.X_test)} rows"
    )
    for sname, sp in season_dict.items():
        print(
            f"  {sname:8s} — train: {len(sp.X_train)}, "
            f"val: {len(sp.X_val)}, "
            f"test: {len(sp.X_test)} rows"
        )

    all_splits: dict[str, SplitData] = {"Full Year": full_split}
    all_splits.update(season_dict)

    # 5. Train and evaluate models
    print("\n" + "=" * 60)
    print("STEP 4: Training and evaluating models")
    print("=" * 60)
    results: dict[str, dict[str, object]] = {}
    best_params_dict: dict[str, dict[str, Any]] = {}
    rf_importances_dict: dict[str, np.ndarray] = {}
    scatter_data: dict[str, dict[str, dict[str, np.ndarray]]] = {"RR": {}, "RF": {}}

    loaded_params: dict[str, dict[str, Any]] = {}
    if not do_grid_search:
        params_path = RESULTS_DIR / "best_hyperparameters.json"
        if params_path.exists():
            with open(params_path, "r") as f:
                loaded_params = json.load(f)
            print(f"  Loaded hyperparameters from {params_path}")

    for split_name, split_data in all_splits.items():
        print(f"\n--- {split_name} ---")
        results[split_name] = {}
        best_params_dict[split_name] = {}

        for model_name in ("RR", "RF"):
            label = "Ridge (RR)" if model_name == "RR" else "Random Forest"
            print(f"  Training {label} ...")
            preloaded_params = None
            if not do_grid_search and split_name in loaded_params and model_name in loaded_params[split_name]:
                preloaded_params = loaded_params[split_name][model_name]

            metrics, fitted, best_params, rf_importance, y_true, y_pred = _run_model_on_split(
                model_name, split_data, split_name, feature_cols,
                do_grid_search=(do_grid_search and model_name == "RF"),
                preloaded_params=preloaded_params,
            )
            results[split_name][model_name] = metrics
            
            if split_name != "Full Year":
                if rf_importance is not None:
                    rf_importances_dict[split_name] = rf_importance
                scatter_data[model_name][split_name] = {"y_true": y_true, "y_pred": y_pred}
            
            if best_params:
                best_params_dict[split_name][model_name] = best_params
                
            print(
                f"    nRMSE={metrics.nrmse:.2f}%  "
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

    if do_grid_search:
        params_path = RESULTS_DIR / "best_hyperparameters.json"
        with open(params_path, "w") as f:
            json.dump(best_params_dict, f, indent=4)
        print(f"  Hyperparameters -> {params_path}")

    # 7. Seasonal bar charts (all four seasons, test-set metrics)
    if not summary_df.empty:
        plot_seasonal_bars(summary_df)
        print("  Seasonal bar charts saved.")

    # 8. Feature importance and scatter subplots
    if rf_importances_dict:
        from src.visualization import plot_seasonal_feature_importances, plot_seasonal_scatter
        plot_seasonal_feature_importances(rf_importances_dict, feature_cols)
        print("  Seasonal feature importances saved.")
        
        for model_name, s_data in scatter_data.items():
            if s_data:
                plot_seasonal_scatter(model_name, s_data)
        print("  Seasonal scatter plots saved.")

    # 9. Console summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (Test Set)")
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
        description="Ridge vs. Random Forest: Solar Power Forecasting (Nordic)",
    )
    parser.add_argument(
        "--production", "-p",
        required=True,
        help="Path to the solar production CSV file (your panel data).",
    )
    parser.add_argument(
        "--temp-station",
        type=int,
        default=SMHI_STATION_TEMP,
        help=f"SMHI station ID for temperature (default: {SMHI_STATION_TEMP}).",
    )
    parser.add_argument(
        "--ghi-station",
        type=int,
        default=SMHI_STATION_GHI,
        help=f"SMHI station ID for GHI/irradiance (default: {SMHI_STATION_GHI}).",
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
        help="Run grid search for RF hyperparameter tuning (uses val set, slower).",
    )
    args = parser.parse_args()

    run_pipeline(
        production_csv=args.production,
        temp_station_id=args.temp_station,
        ghi_station_id=args.ghi_station,
        smhi_csv=args.smhi_csv,
        do_grid_search=args.grid_search,
    )


if __name__ == "__main__":
    main()
