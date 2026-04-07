"""Configuration constants for the WB hack pipeline."""

TRACK = "team"
RANDOM_STATE = 42

TRACK_CONFIG = {
    "solo": {
        "train_path": "train_solo_track.parquet",
        "test_path": "test_solo_track.parquet",
        "target_col": "target_1h",
        "forecast_points": 8,
    },
    "team": {
        "train_path": "train_team_track.parquet",
        "test_path": "test_team_track.parquet",
        "target_col": "target_2h",
        "forecast_points": 10,
    },
}

CONFIG = TRACK_CONFIG[TRACK]
TARGET_COL = CONFIG["target_col"]
FORECAST_POINTS = CONFIG["forecast_points"]
FUTURE_TARGET_COLS = [f"target_step_{step}" for step in range(1, FORECAST_POINTS + 1)]

# --- Training params ---
TRAIN_DAYS = 21
MAX_TRAIN_ROWS = 2_000_000
VALID_FRAC = 0.8  # time-based split fraction

# --- LightGBM defaults ---
LGB_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}
