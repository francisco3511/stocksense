from pathlib import Path

# Base paths
ROOT_DIR = Path(__file__).parents[2]
PACKAGE_DIR = ROOT_DIR / "stocksense"

# Data paths
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

# Database paths
DATABASE_DIR = DATA_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "stock_db.db"

# Model paths
MODEL_DIR = ROOT_DIR / "models"

# Report paths
REPORTS_DIR = ROOT_DIR / "reports"
SCORES_DIR = REPORTS_DIR / "scores"
PORTFOLIO_DIR = REPORTS_DIR / "portfolios"

# Test paths
TEST_DIR = ROOT_DIR / "tests"
FIXTURES_DIR = TEST_DIR / "fixtures"

# Ensure required directories exist
REQUIRED_DIRS = [
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    CACHE_DIR,
    DATABASE_DIR,
    MODEL_DIR,
    SCORES_DIR,
    PORTFOLIO_DIR,
    FIXTURES_DIR,
]

for directory in REQUIRED_DIRS:
    directory.mkdir(parents=True, exist_ok=True)
