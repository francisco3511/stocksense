import os
from pathlib import Path

# disable DeprecationWarning https://github.com/jupyter/jupyter_core/issues/398
os.environ["JUPYTER_PLATFORM_DIRS"] = "1"

PROJECT_DIR = Path(__file__).parents[2]
PACKAGE_DIR = PROJECT_DIR / "stocksense"
FIXTURES_DIR = Path(__file__).parent / "fixtures"
OUTPUT_DIR = PACKAGE_DIR / "data"
