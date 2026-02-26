"""
config.py
---------
Central configuration for the solar power forecasting pipeline.
All paths, feature definitions, seasonal boundaries, model hyperparameters,
SMHI API settings, and site-specific constants are defined here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SMHI_CACHE_DIR = DATA_DIR / "smhi_cache"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# SMHI Open Data API
# ---------------------------------------------------------------------------
SMHI_BASE_URL = (
    "https://opendata-download-metobs.smhi.se/api/version/1.0"
)
SMHI_PARAM_TEMPERATURE = 1   # Lufttemperatur, momentanvärde, 1 gång/tim (°C)
SMHI_PARAM_GHI = 11          # Global Irradians, medelvärde 1 timma (W/m²)

# Default stations (Stockholm area — change to match your panels' location).
# SMHI uses separate station IDs for temperature and GHI (solar irradiance).
# Temperature: Stockholm-Observatoriekullen A
SMHI_STATION_TEMP = 98230
# GHI: Stockholm Sol
SMHI_STATION_GHI = 98735

# ---------------------------------------------------------------------------
# Production CSV Column Names
# ---------------------------------------------------------------------------
# Adjust these to match the column names in YOUR solar production CSV.
PRODUCTION_TIMESTAMP_COL = "timestamp"
PRODUCTION_POWER_COL = "power_kw"

# ---------------------------------------------------------------------------
# Internal Column Names (used after loading/merging)
# ---------------------------------------------------------------------------
TARGET_COL = "power_kw"

# Raw meteorological feature columns (from SMHI)
RAW_FEATURE_COLS = [
    "temperature",   # Ambient temperature (°C)
    "ghi",           # Global Horizontal Irradiance (W/m²)
]

# ---------------------------------------------------------------------------
# Seasonal Definitions (Nordic calendar seasons)
# ---------------------------------------------------------------------------
SEASONS = {
    "Winter": (12, 1, 2),
    "Spring": (3, 4, 5),
    "Summer": (6, 7, 8),
    "Autumn": (9, 10, 11),
}

# Mapping: month number -> season name (convenience lookup)
MONTH_TO_SEASON = {}
for _season_name, _months in SEASONS.items():
    for _m in _months:
        MONTH_TO_SEASON[_m] = _season_name

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
# Temperature coefficient for crystalline-silicon PV (per °C)
TEMP_COEFFICIENT_GAMMA = -0.004  # typ. range: -0.003 to -0.005

# Standard Test Conditions temperature (°C)
STC_TEMPERATURE = 25.0

# Snow-cover proxy: temperature threshold and eligible months
SNOW_TEMP_THRESHOLD_C = 2.0
SNOW_ELIGIBLE_MONTHS = (11, 12, 1, 2, 3)

# Solar constant for extraterrestrial irradiance approximation (W/m²)
SOLAR_CONSTANT = 1361.0

# ---------------------------------------------------------------------------
# Model Hyperparameters
# ---------------------------------------------------------------------------
# Random Forest grid for GridSearchCV
RF_PARAM_GRID = {
    "rf__n_estimators": [100, 300, 500],
    "rf__max_depth": [10, 20, None],
    "rf__min_samples_split": [2, 5],
    "rf__min_samples_leaf": [1, 4],
}

# TimeSeriesSplit folds for cross-validation
CV_N_SPLITS = 5

# ---------------------------------------------------------------------------
# Train / Test split ratio (chronological)
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.80

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
PLOT_STYLE = "seaborn-v0_8-whitegrid"
