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

    def __init__(self, db_path: str = 'data/database/stock_db.db'):
        self.db = DatabaseConnection(Path(db_path))
        create_tables(self.db.get_connection())

    def insert_stock(self, data: pl.DataFrame):
        data = data.with_columns(
            data['last_update'].dt.strftime('%Y-%m-%d'),
        )
        insert_data(self.db.get_connection(), "stock", data)

    def insert_info(self, record: dict):
        insert_record(self.db.get_connection(), "info", record)

    def insert_market_data(self, data: pl.DataFrame):
        data = data.with_columns(data['date'].dt.strftime('%Y-%m-%d').alias('date'))
        insert_data(self.db.get_connection(), "market", data)

    def insert_financial_data(self, data: pl.DataFrame):
        data = data.with_columns(
            data['rdq'].dt.strftime('%Y-%m-%d'),
            data['datadate'].dt.strftime('%Y-%m-%d')
        )
        insert_data(self.db.get_connection(), "financial", data)

    def insert_insider_data(self, data: pl.DataFrame):
        data = data.with_columns(
            data['filling_date'].dt.strftime('%Y-%m-%d'),
            data['trade_date'].dt.strftime('%Y-%m-%d')
        )
        insert_data(self.db.get_connection(), "insider", data)

    def insert_index_data(self, data: pl.DataFrame):
        data = data.with_columns(data['date'].dt.strftime('%Y-%m-%d'))
        insert_data(self.db.get_connection(), "sp500", data)
        
    def delete_stock(self, tic: str):
        delete_data(self.db.get_connection(), "stock", {"tic": tic})

    def update_stock(self, tic: str, update_values: dict):
        update_data(
            self.db.get_connection(),
            "stock",
            update_values,
            {"tic": tic}
        )

    def count_stocks(self) -> int:
        return count_data(self.db.get_connection(), "stock", "tic")

    def fetch_stock(
        self,
        tic: Optional[int] = None
    ) -> Union[pl.DataFrame, tuple]:
        
        # connect to db and fetch data
        conn = self.db.get_connection()
        df = fetch_data(conn, "stock", {"tic": tic} if tic else None)
        
        if df.is_empty():
            return df

        return df.with_columns(
            pl.col('last_update').str.to_date(format='%Y-%m-%d')
        )
 
    def fetch_info(
        self,
        tic: Optional[int] = None
    ) -> pl.DataFrame:
        
        # connect to db and fetch data
        conn = self.db.get_connection()
                
        # Fetch record for a single stock or all stocks
        return (
            fetch_record(conn, "info", {"tic": tic}) if tic 
            else fetch_data(conn, "info")
        )
    
    def fetch_market_data(
        self,
        tic: Optional[int] = None
    ) -> pl.DataFrame:
        
        # connect to db and fetch data
        conn = self.db.get_connection()
        df = fetch_data(conn, "market", {"tic": tic} if tic else None)

        if df.is_empty():
            return df

        # format dates
        return df.with_columns(
            pl.col('date').str.to_date(format='%Y-%m-%d')
        )

    def fetch_financial_data(
        self,
        tic: Optional[int] = None
    ) -> pl.DataFrame:
        
        # connect to db and fetch data
        conn = self.db.get_connection()
        df = fetch_data(conn, "financial", {"tic": tic} if tic else None)

        if df.is_empty():
            return df

        # format dates
        return df.with_columns([
            pl.col('datadate').str.to_date(format='%Y-%m-%d'),
            pl.col('rdq').str.to_date(format='%Y-%m-%d')
        ])

    def fetch_insider_data(
        self,
        tic: Optional[int] = None
    ) -> pl.DataFrame:
        
        # connect to db and fetch data
        conn = self.db.get_connection()
        df = fetch_data(conn, "insider", {"tic": tic} if tic else None)
        
        if df.is_empty():
            return df
        
        # format dates
        return df.with_columns([
            pl.col('trade_date').str.to_date(format='%Y-%m-%d'),
            pl.col('filling_date').str.to_date(format='%Y-%m-%d')
        ])

    def fetch_sp_data(self) -> pl.DataFrame:
        return fetch_data(self.db.get_connection(), "sp500")

    def close(self):
        self.db.close()
