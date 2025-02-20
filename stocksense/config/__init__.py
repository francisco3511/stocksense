from .manager import ConfigManager
from .paths import (
    CACHE_DIR,
    DATA_DIR,
    DATABASE_DIR,
    DATABASE_PATH,
    FIXTURES_DIR,
    INTERIM_DATA_DIR,
    MODEL_DIR,
    PACKAGE_DIR,
    PORTFOLIO_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    ROOT_DIR,
    SCORES_DIR,
    TEST_DIR,
)

config = ConfigManager()

__all__ = [
    "config",
    "CACHE_DIR",
    "DATA_DIR",
    "DATABASE_DIR",
    "DATABASE_PATH",
    "FIXTURES_DIR",
    "INTERIM_DATA_DIR",
    "MODEL_DIR",
    "PACKAGE_DIR",
    "PORTFOLIO_DIR",
    "PROCESSED_DATA_DIR",
    "RAW_DATA_DIR",
    "REPORTS_DIR",
    "ROOT_DIR",
    "SCORES_DIR",
    "TEST_DIR",
]
