import datetime as dt
import sqlite3
from typing import Optional

import numpy as np
import polars as pl

from .connection import DatabaseConnection
from .queries import (
    count_data,
    delete_data,
    delete_table,
    fetch_data,
    insert_data,
    insert_record,
    update_data,
)
from .schema import create_tables

sqlite3.register_adapter(np.int32, int)
sqlite3.register_adapter(np.int64, int)


class DatabaseHandler:
    """
    Wrapper for database handling.
    """

    def __init__(self):
        self.db = DatabaseConnection()
        conn = self.db.get_connection()
        create_tables(conn)

    def insert_stock(self, data: pl.DataFrame) -> None:
        data = convert_date_columns_to_str(data, ["date_added", "date_removed"])
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

    def insert_vix_data(self, data: pl.DataFrame) -> None:
        data = convert_date_columns_to_str(data, ["date"])
        conn = self.db.get_connection()
        insert_data(conn, "vix", data)

    def delete_stock(self, tic: str) -> None:
        conn = self.db.get_connection()
        delete_data(conn, "stock", {"tic": tic})

    def delete_financials(self, tic: str) -> None:
        conn = self.db.get_connection()
        delete_data(conn, "financial", {"tic": tic})

    def delete_market_data(self, tic: str) -> None:
        conn = self.db.get_connection()
        delete_data(conn, "market", {"tic": tic})

    def update_stock(self, tic: str, update_values: dict) -> None:
        conn = self.db.get_connection()
        update_data(conn, "stock", update_values, {"tic": tic})

    def count_stocks(self) -> int | None:
        conn = self.db.get_connection()
        return count_data(conn, "stock", "tic")

    def fetch_stock(self, tic: Optional[str] = None) -> pl.DataFrame:
        conn = self.db.get_connection()
        df = fetch_data(conn, "stock")
        if df is None:
            return pl.DataFrame()
        df = convert_str_columns_to_date(df, ["date_added", "date_removed"])
        return df.filter(pl.col("tic") == tic) if tic else df

    def fetch_constituents(self, trade_date: dt.datetime) -> list[str]:
        """Get constituents for a given trade date."""
        stocks = self.fetch_stock()
        constituents = stocks.filter(
            (pl.col("date_removed").is_null() | (pl.col("date_removed") > trade_date))
            & (pl.col("date_added").is_null() | (pl.col("date_added") <= trade_date))
        )["tic"].to_list()
        return constituents

    def fetch_sp500_stocks(self) -> list[str]:
        conn = self.db.get_connection()
        df = fetch_data(conn, "stock")
        if df is None:
            return []
        return df.filter(pl.col("date") == 1)["tic"].to_list()

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

    def fetch_vix_data(self) -> pl.DataFrame:
        conn = self.db.get_connection()
        df = fetch_data(conn, "vix")
        if df is None:
            return pl.DataFrame()
        df = convert_str_columns_to_date(df, ["date"])
        return df

    def delete_table(self, table_name: str) -> bool:
        conn = self.db.get_connection()
        return delete_table(conn, table_name)

    def close(self):
        self.db.close()


def convert_date_columns_to_str(df, cols, date_format="%Y-%m-%d") -> pl.DataFrame:
    return df.with_columns([df[col].cast(pl.Date).dt.strftime(date_format) for col in cols])


def convert_str_columns_to_date(df, cols, date_format="%Y-%m-%d") -> pl.DataFrame:
    return df.with_columns(
        [pl.col(col).str.to_date(format=date_format, strict=False) for col in cols]
    )
