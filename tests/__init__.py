from pathlib import Path

PROJECT_DIR = Path(__file__).parents[2]
PACKAGE_DIR = PROJECT_DIR / "stocksense"
FIXTURES_DIR = Path(__file__).parent / "fixtures"
OUTPUT_DIR = PACKAGE_DIR / "data"
