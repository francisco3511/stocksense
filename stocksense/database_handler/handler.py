import numpy as np
from pathlib import Path
import polars as pl
import sqlite3
from typing import Optional, Union

from database_handler import (
    DatabaseConnection,
    create_tables,
    insert_data,
    insert_record,
    update_data,
    delete_data,
    count_data,
    fetch_record,
    fetch_data
)

sqlite3.register_adapter(np.int32, int)
sqlite3.register_adapter(np.int64, int)


class DatabaseHandler:
    """
    Wrapper for database handling.
    """

    def __init__(self, db_path: str = 'data/database/stocks.db'):
        self.db = DatabaseConnection(Path(db_path))
        create_tables(self.db.get_connection())

    def insert_stock_info(self, data: pl.DataFrame):
        data = data.with_columns(
            data['last_update'].dt.strftime('%Y-%m-%d'),
        )
        insert_data(self.db.get_connection(), "stocks", data)

    def insert_metadata(self, record: dict):
        insert_record(self.db.get_connection(), "metadata", record)

    def insert_market_data(self, data: pl.DataFrame):
        data = data.with_columns(data['date'].dt.strftime('%Y-%m-%d').alias('date'))
        insert_data(self.db.get_connection(), "market", data)

    def insert_financial_data(self, data: pl.DataFrame):
        data = data.with_columns(
            data['rdq'].dt.strftime('%Y-%m-%d'),
            data['datadate'].dt.strftime('%Y-%m-%d')
        )
        insert_data(self.db.get_connection(), "financials", data)

    def insert_insider_data(self, data: pl.DataFrame):
        data = data.with_columns(
            data['filling_date'].dt.strftime('%Y-%m-%d'),
            data['trade_date'].dt.strftime('%Y-%m-%d')
        )
        insert_data(self.db.get_connection(), "insider", data)

    def insert_sp_data(self, data: pl.DataFrame):
        data = data.with_columns(data['date'].dt.strftime('%Y-%m-%d'))
        insert_data(self.db.get_connection(), "sp500", data)
        
    def delete_stock_data(self, tic: str):
        delete_data(self.db.get_connection(), "stocks", {"tic": tic})

    def update_stock_data(self, tic: str, update_values: dict):
        update_data(
            self.db.get_connection(),
            "stocks",
            update_values,
            {"tic": tic}
        )

    def count_stocks(self) -> int:
        return count_data(self.db.get_connection(), "stocks", "tic")

    def fetch_stock_info(
        self,
        tic: Optional[int] = None
    ) -> Union[pl.DataFrame, tuple]:
        
        conn = self.db.get_connection()
        
        if tic:
            df = fetch_data(conn, "stocks", {"tic": tic})
        else:
            df = fetch_data(self.db.get_connection(), "stocks")
        
        df = df.with_columns(
            pl.col('last_update').str.to_date(format='%Y-%m-%d')
        )
        return df

    def fetch_metadata(
        self,
        tic: Optional[int] = None
    ) -> pl.DataFrame:
        
        if tic:
            # fetch metadata record of a single stock
            return fetch_record(self.db.get_connection(), "metadata", {"tic": tic})
        else:
            # fetch all metadata
            return fetch_data(self.db.get_connection(), "metadata")

    def fetch_market_data(
        self,
        tic: Optional[int] = None
    ) -> pl.DataFrame:
        
        conn = self.db.get_connection()
        
        if tic:
            df = fetch_data(conn, "market", {"tic": tic})
        else:
            df = fetch_data(conn, "market")
        
        # format dates
        df = df.with_columns(
            pl.col('date').str.to_date(format='%Y-%m-%d')
        )
        
        return df

    def fetch_financial_data(
        self,
        tic: Optional[int] = None
    ) -> pl.DataFrame:
        
        if tic:
            df = fetch_data(self.db.get_connection(), "financials", {"tic": tic})
        else:
            df = fetch_data(self.db.get_connection(), "financials")

        # format dates
        df = df.with_columns([
            pl.col('datadate').str.to_date(format='%Y-%m-%d'),
            pl.col('rdq').str.to_date(format='%Y-%m-%d')
        ])
        
        return df

    def fetch_insider_data(
        self,
        tic: Optional[int] = None
    ) -> pl.DataFrame:
        
        if tic:
            # fetch single stock
            df = fetch_data(self.db.get_connection(), "insider", {"tic": tic})
        else:
            # fetch all insider trading records
            df = fetch_data(self.db.get_connection(), "insider")

        # format dates
        df = df.with_columns([
            pl.col('trade_date').str.to_date(format='%Y-%m-%d'),
            pl.col('filling_date').str.to_date(format='%Y-%m-%d')
        ])

        return df

    def fetch_sp_data(self) -> pl.DataFrame:
        return fetch_data(self.db.get_connection(), "sp500")

    def close(self):
        self.db.close()
