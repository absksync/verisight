import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BACKEND_DIR / "uploads"
PROCESSED_DIR = DATASETS_DIR / "processed"

# Ensure runtime directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Raw Datasets Paths
EXPDATE_DIR = DATASETS_DIR / "raw" / "expdate"
FOODPACKAGING_DIR = DATASETS_DIR / "raw" / "foodpackagingocr"
IMD2020_DIR = DATASETS_DIR / "raw" / "imd2020"
OPENFOODFACTS_DIR = DATASETS_DIR / "raw" / "openfoodfacts"

# Application Settings
API_PREFIX = "/api"
PROJECT_NAME = "VeriSight - Expiry & Packaging Integrity AI"
VERSION = "1.0.0"

# Forensic / Tampering Detection Parameters
ELA_QUALITY = 90
ELA_SCALE = 15.0
TAMPER_ANOMALY_THRESHOLD = 0.45  # Scores > 0.45 are suspicious, > 0.70 are tampered

# Expiry Date Thresholds
EXPIRING_SOON_DAYS = 30  # Days threshold to trigger warning status
DEFAULT_SHELF_LIFE_DAYS = 365
