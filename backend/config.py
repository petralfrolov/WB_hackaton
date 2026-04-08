from __future__ import annotations

import os
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parent
DATA_DIR = BACKEND_DIR / "data"
MODELS_DIR = BACKEND_DIR / "models"

APP_TITLE = "WB Transport Optimizer"
APP_VERSION = "1.0.0"
API_PORT = 8000

ALLOWED_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

READ_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
DISPATCH_PATHS = {"/dispatch", "/optimize", "/call"}

DB_FILENAME = os.getenv("WB_DB_FILENAME", "app.db")
DB_PATH = DATA_DIR / DB_FILENAME
DB_CONNECT_ARGS = {"check_same_thread": False}
DB_ECHO = False

DEFAULT_TRAIN_PATH = DATA_DIR / "train_team_track.parquet"
DEFAULT_MODELS_DIR = MODELS_DIR / "models"
DEFAULT_DEPRECATED_MODELS_DIR = MODELS_DIR / "exp_best_features_full"

NCS_DEFAULT_PATH = DATA_DIR / "non_conformity_scores.csv"
NCS_NORM_PATH = DATA_DIR / "non_conformity_scores_norm.csv"
NCS_ALLSTEPS_PATH = DATA_DIR / "non_conformity_scores_norm_allsteps.csv"
CONFORMAL_HORIZONS = ("0-2h", "2-4h", "4-6h")

DISPATCH_HORIZONS = [
    ("A: now", 0),
    ("B: +2h", 120),
    ("C: +4h", 240),
    ("D: +6h", 360),
]
DISPATCH_HORIZON_LABELS = [label for label, _ in DISPATCH_HORIZONS]
ALLSTEP_HORIZON_LABELS = [
    "-1.5-0.5h", "-1-1h", "-0.5-1.5h", "0-2h",
    "0.5-2.5h", "1-3h", "1.5-3.5h", "2-4h",
    "2.5-4.5h", "3-5h", "3.5-5.5h", "4-6h",
]

DEFAULT_PERIOD_MINUTES = 120
DB_PERIOD_MINUTES = 30
DISPATCH_WINDOW_MINUTES = 360
DEFAULT_SOLVER_TIME_LIMIT_SECONDS = 30.0
DEFAULT_ROUTE_DISTANCE_KM = 15.0
DEFAULT_FLEET_AVAILABLE = 1000

DEFAULT_WAIT_PENALTY_PER_MINUTE = 8.0
DEFAULT_ECONOMY_THRESHOLD = 0.0
DEFAULT_CONFIDENCE_LEVEL = 0.9
DEFAULT_ROUTE_CORRELATION = 0.3
DEFAULT_GRANULARITY_HOURS = 2.0
DEFAULT_DISPATCH_CONCURRENCY = 3

TRACK_NAME = "team"
TARGET_COLUMN = "target_2h"
FORECAST_POINTS = 12
HALFHOUR_SLOT_MINUTES = 30
MAX_LAG_STEPS = 672
LOOKBACK_HEADROOM_DAYS = 2