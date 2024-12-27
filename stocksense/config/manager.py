from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)


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


class ScrapingConfig(BaseModel):
    base_date: str = Field(description="Starting date for data collection in YYYY-MM-DD format")
    crsp_columns: List[str] = Field(min_length=1)
    macrotrends: Dict[str, Dict[str, str]]
    yahoo: Dict[str, str]
    yahoo_info: Dict[str, str]

    @property
    def start_date(self) -> datetime:
        """Convert base_date string to datetime object."""
        return datetime.strptime(self.base_date, "%Y-%m-%d")


class ProcessingConfig(BaseModel):
    trade_days_2week: int = Field(alias="two_week_trading_days")
    trade_days_month: int = Field(alias="month_trading_days")
    trade_days_quarter: int = Field(alias="quarter_trading_days")
    trade_days_semester: int = Field(alias="semester_trading_days")
    trade_days_third_quarter: int = Field(alias="third_quarter_trading_days")
    trade_days_year: int = Field(alias="year_trading_days")
    trade_days_2year: int = Field(alias="two_year_trading_days")
    prediction_horizon: int = Field(gt=0)
    over_performance_threshold: float = Field(lt=1.0, description="Overperformance threshold")
    performance_threshold: float = Field(gt=0.0, lt=1.0, description="Performance threshold")
    sectors: List[Sector] = Field(min_length=11, max_length=11)

    @model_validator(mode="after")
    def validate_trading_days_order(self) -> "ProcessingConfig":
        """Validate that trading days are in ascending order."""
        if not (
            self.trade_days_2week
            < self.trade_days_month
            < self.trade_days_quarter
            < self.trade_days_semester
            < self.trade_days_year
            < self.trade_days_2year
        ):
            raise ValueError("Trading days must be in ascending order")
        return self

    @property
    def trading_days(self) -> Dict[str, int]:
        """Get all trading day periods in a dictionary."""
        return {
            "2week": self.trade_days_2week,
            "month": self.trade_days_month,
            "quarter": self.trade_days_quarter,
            "semester": self.trade_days_semester,
            "year": self.trade_days_year,
            "2year": self.trade_days_2year,
        }

    @property
    def performance_thresholds(self) -> Dict[str, float]:
        """Get all performance thresholds in a dictionary."""
        return {
            "over_performance": self.over_performance_threshold,
            "performance": self.performance_threshold,
        }

    @model_validator(mode="after")
    def validate_sectors_completeness(self) -> "ProcessingConfig":
        """Validate that all sectors are present."""
        if set(self.sectors) != set(Sector):
            missing = set(Sector) - set(self.sectors)
            raise ValueError(f"Missing sectors: {missing}")
        return self


class ModelConfig(BaseModel):
    features: List[str]
    targets: List[str]
    id_col: str
    date_col: str
    min_train_years: int = Field(ge=5, le=50)
    ga: Dict[str, Any]

    @model_validator(mode="after")
    def validate_column_names(self) -> "ModelConfig":
        """Validate that target, id_col and date_col are not in features."""
        special_cols = [self.targets] + [self.id_col, self.date_col]
        if any(col in self.features for col in special_cols):
            raise ValueError("features list cannot contain target, id_col or date_col")
        return self

    @property
    def feature_count(self) -> int:
        """Get the number of features."""
        return len(self.features)

    @property
    def window_sizes(self) -> Dict[str, int]:
        """Get all window sizes in a dictionary."""
        return {
            "train": self.train_window,
            "validation": self.val_window,
        }


class DatabaseConfig(BaseModel):
    db_schema: Dict[str, List[str]]

    @field_validator("db_schema")
    @classmethod
    def validate_schema_structure(cls, v: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Validate that each table has at least one column."""
        for table, columns in v.items():
            if not columns:
                raise ValueError(f"Table '{table}' must have at least one column")
        return v

    @property
    def tables(self) -> List[str]:
        """Get list of all tables in the database."""
        return list(self.db_schema.keys())

    def get_columns(self, table: str) -> List[str]:
        """Get columns for a specific table."""
        if table not in self.db_schema:
            raise ValueError(f"Table '{table}' not found in schema")
        return self.db_schema[table]


class ConfigManager:
    def __init__(self):
        self.config_path = Path(__file__).parent / "defaults"
        self.scraping: ScrapingConfig = None
        self.processing: ProcessingConfig = None
        self.model: ModelConfig = None
        self.database: DatabaseConfig = None
        self._load_configs()

    def _load_configs(self):
        self.scraping = ScrapingConfig(**self._get_config("scraping"))
        self.processing = ProcessingConfig(**self._get_config("processing"))
        self.model = ModelConfig(**self._get_config("model"))
        self.database = DatabaseConfig(**self._get_config("db"))

    def _get_config(self, config_file: str) -> dict:
        config_path = self.config_path / f"{config_file}_config.yml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, encoding="utf8") as file:
            try:
                return yaml.safe_load(file)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML in {config_file}_config.yml: {e}") from e

    @property
    def trading_days(self) -> Dict[str, int]:
        """Shortcut to access trading days configuration."""
        return self.processing.trading_days

    @property
    def features(self) -> List[str]:
        """Shortcut to access model features."""
        return self.model.features

    @property
    def tables(self) -> List[str]:
        """Shortcut to access database tables."""
        return self.database.tables

    def get_table_columns(self, table: str) -> List[str]:
        """Shortcut to get columns for a specific table."""
        return self.database.get_columns(table)
