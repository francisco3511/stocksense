import numpy as np
from pathlib import Path
import polars as pl
import sqlite3
from typing import Optional

from database_handler import (
    DatabaseConnection,
    create_tables,
    insert_data,
    insert_record,
    update_data,
    delete_data,
    count_data,
    fetch_data,
)

sqlite3.register_adapter(np.int32, int)
sqlite3.register_adapter(np.int64, int)


class DatabaseHandler:
    """
    Wrapper for database handling.
    """

    def __init__(self, db_path: str = "data/database/stock_db.db"):
        self.db = DatabaseConnection(Path(db_path))
        conn = self.db.get_connection()
        create_tables(conn)

    def insert_stock(self, data: pl.DataFrame) -> None:
        data = convert_date_columns_to_str(data, ["last_update"])
        insert_data(self.db.get_connection(), "stock", data)

    def insert_info(self, record: dict) -> None:
        insert_record(self.db.get_connection(), "info", record)

    def insert_market_data(self, data: pl.DataFrame) -> None:
        data = convert_date_columns_to_str(data, ["date"])
        conn = self.db.get_connection()
        insert_data(conn, "market", data)

    def insert_financial_data(self, data: pl.DataFrame) -> None:
        data = convert_date_columns_to_str(data, ["rdq", "datadate"])
        conn = self.db.get_connection()
        insert_data(conn, "financial", data)

    def insert_insider_data(self, data: pl.DataFrame) -> None:
        data = convert_date_columns_to_str(data, ["filling_date", "trade_date"])
        conn = self.db.get_connection()
        insert_data(conn, "insider", data)

    def insert_index_data(self, data: pl.DataFrame) -> None:
        data = convert_date_columns_to_str(data, ["date"])
        conn = self.db.get_connection()
        insert_data(conn, "sp500", data)

    def delete_stock(self, tic: str) -> None:
        conn = self.db.get_connection()
        delete_data(conn, "stock", {"tic": tic})

    def delete_financials(self, tic: str) -> None:
        conn = self.db.get_connection()
        delete_data(conn, "financial", {"tic": tic})

    def update_stock(self, tic: str, update_values: dict) -> None:
        conn = self.db.get_connection()
        update_data(conn, "stock", update_values, {"tic": tic})

    def count_stocks(self) -> int | None:
        conn = self.db.get_connection()
        return count_data(conn, "stock", "tic")

    def fetch_stock(self, tic: Optional[str] = None) -> pl.DataFrame:
        conn = self.db.get_connection()
        df = fetch_data(conn, "stock", {"tic": tic} if tic else None)
        if df is None:
            return pl.DataFrame()
        df = convert_str_columns_to_date(df, ["last_update"])
        return df

    def fetch_info(self, tic: Optional[str] = None) -> pl.DataFrame:
        conn = self.db.get_connection()
        df = fetch_data(conn, "info", {"tic": tic} if tic else None)
        if df is None:
            return pl.DataFrame()
        return df

    def fetch_market_data(self, tic: Optional[str] = None) -> pl.DataFrame:
        conn = self.db.get_connection()
        df = fetch_data(conn, "market", {"tic": tic} if tic else None)

        if df is None:
            return pl.DataFrame()

        df = convert_str_columns_to_date(df, ["date"])
        return df

    def fetch_financial_data(self, tic: Optional[str] = None) -> pl.DataFrame:
        conn = self.db.get_connection()
        df = fetch_data(conn, "financial", {"tic": tic} if tic else None)
        if df is None:
            return pl.DataFrame()
        df = convert_str_columns_to_date(df, ["datadate", "rdq"])
        return df

    def fetch_insider_data(self, tic: Optional[str] = None) -> pl.DataFrame:
        conn = self.db.get_connection()
        df = fetch_data(conn, "insider", {"tic": tic} if tic else None)
        if df is None:
            return pl.DataFrame()
        df = convert_str_columns_to_date(df, ["filling_date", "trade_date"])
        return df

    def fetch_index_data(self) -> pl.DataFrame:
        conn = self.db.get_connection()
        df = fetch_data(conn, "sp500")
        if df is None:
            return pl.DataFrame()
        df = convert_str_columns_to_date(df, ["date"])
        return df

    def close(self):
        self.db.close()


def convert_date_columns_to_str(df, cols, date_format="%Y-%m-%d") -> pl.DataFrame:
    return df.with_columns([df[col].dt.strftime(date_format) for col in cols])


def convert_str_columns_to_date(df, cols, date_format="%Y-%m-%d") -> pl.DataFrame:
    return df.with_columns(
        [pl.col(col).str.to_date(format=date_format) for col in cols]
    )
