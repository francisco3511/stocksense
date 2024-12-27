from datetime import date
from enum import Enum

import pandera.polars as pa
import polars as pl


class BaseSchema(pa.DataFrameModel):
    class Config:
        coerce = True
        strict = False


class Sector(str, Enum):
    HEALTH_CARE = "Health Care"
    FINANCIALS = "Financials"
    INDUSTRIALS = "Industrials"
    CONSUMER_DISCRETIONARY = "Consumer Discretionary"
    INFORMATION_TECHNOLOGY = "Information Technology"
    COMMUNICATION_SERVICES = "Communication Services"
    CONSUMER_STAPLES = "Consumer Staples"
    UTILITIES = "Utilities"
    REAL_ESTATE = "Real Estate"
    MATERIALS = "Materials"
    ENERGY = "Energy"


def validate_df(schema: pa.DataFrameModel, df: pl.DataFrame) -> pl.DataFrame:
    """Validate a polars DataFrame using a pandera schema."""
    date_columns = [col for col in df.columns if col == "date"]
    renamed_df = df.rename({col: f"table_{col}" for col in date_columns})
    validated_df = schema.validate(renamed_df)
    return validated_df.rename({f"table_{col}": col for col in date_columns})


class StockSchema(BaseSchema):
    tic: str = pa.Field(nullable=False, unique=True)
    name: str = pa.Field(nullable=True)
    sector: str = pa.Field(nullable=False, isin=[s.value for s in Sector])
    date_added: date = pa.Field(nullable=True)
    date_removed: date = pa.Field(nullable=True)


class FinancialSchema(BaseSchema):
    tic: str = pa.Field(nullable=False)
    datadate: date = pa.Field(nullable=False)
    rdq: date = pa.Field(nullable=False)
    saleq: float = pa.Field(nullable=True, ge=0)
    cogsq: float = pa.Field(nullable=True, ge=0)
    xsgaq: float = pa.Field(nullable=True, ge=0)
    niq: float = pa.Field(nullable=True)
    ebitdaq: float = pa.Field(nullable=True)
    cshoq: float = pa.Field(nullable=True, ge=0)
    atq: float = pa.Field(nullable=True, ge=0)
    cheq: float = pa.Field(nullable=True, ge=0)
    ltq: float = pa.Field(nullable=True, ge=0)
    seqq: float = pa.Field(nullable=True)
    oancfq: float = pa.Field(nullable=True)
    surprise_pct: float = pa.Field(nullable=True, in_range={"min_value": -100, "max_value": 100})


class MarketSchema(BaseSchema):
    tic: str = pa.Field(nullable=False)
    table_date: date = pa.Field(nullable=False)
    close: float = pa.Field(nullable=False, ge=0)
    adj_close: float = pa.Field(nullable=False, ge=0)
    volume: int = pa.Field(nullable=False, ge=0)


class InsiderSchema(BaseSchema):
    tic: str = pa.Field(nullable=False)
    filling_date: date = pa.Field(nullable=False)
    trade_date: date = pa.Field(nullable=False)
    owner_name: str = pa.Field(nullable=False)
    title: str = pa.Field(nullable=True)
    transaction_type: str = pa.Field(
        nullable=False, isin=["P - Purchase", "S - Sale", "S - Sale+OE"]
    )
    qty: int = pa.Field(nullable=False)
    value: str = pa.Field(nullable=False, regex=r"^\$?[\d,]+$")


class SP500Schema(BaseSchema):
    table_date: date = pa.Field(nullable=False)
    close: float = pa.Field(nullable=False, ge=0)
    volume: int = pa.Field(nullable=True, ge=0)


class VIXSchema(BaseSchema):
    table_date: date = pa.Field(nullable=False)
    close: float = pa.Field(nullable=False, ge=0)
