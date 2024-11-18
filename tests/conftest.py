import datetime as dt

import polars as pl
import pytest

from . import FIXTURES_DIR


@pytest.fixture
def info_data() -> pl.DataFrame:
    return pl.read_parquet(FIXTURES_DIR / "stock_data.parquet")


@pytest.fixture
def financial_data() -> pl.DataFrame:
    return pl.read_parquet(FIXTURES_DIR / "financial_data.parquet")


@pytest.fixture
def insider_data() -> pl.DataFrame:
    return pl.read_parquet(FIXTURES_DIR / "insider_data.parquet")


@pytest.fixture
def market_data() -> pl.DataFrame:
    return pl.read_parquet(FIXTURES_DIR / "market_data.parquet")


@pytest.fixture
def index_data() -> pl.DataFrame:
    return pl.read_parquet(FIXTURES_DIR / "index_data.parquet")


@pytest.fixture
def vix_data() -> pl.DataFrame:
    return pl.read_parquet(FIXTURES_DIR / "vix_data.parquet")


@pytest.fixture
def processed_data() -> pl.DataFrame:
    return pl.read_parquet(FIXTURES_DIR / "processed_data.parquet")


@pytest.fixture
def cleaned_data() -> pl.DataFrame:
    return pl.read_parquet(FIXTURES_DIR / "cleaned_data.parquet")


@pytest.fixture
def mock_stock_data():
    return pl.DataFrame(
        {
            "tic": ["AAPL", "OLD", "EXISTING"],
            "name": ["Apple Inc.", "Old Corp", "Existing Corp"],
            "sector": ["Technology", "Finance", "Healthcare"],
            "last_update": [dt.datetime.now().date()] * 3,
            "spx_status": [1, 1, 1],
            "active": [1, 1, 1],
        }
    )


@pytest.fixture
def mock_active_data():
    return pl.DataFrame(
        {
            "tic": ["AAPL", "NEW", "EXISTING"],
            "name": ["Apple Inc.", "New Corp", "Existing Corp"],
            "sector": ["Technology", "Finance", "Healthcare"],
        }
    )
