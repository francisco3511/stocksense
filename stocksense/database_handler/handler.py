import numpy as np
from pathlib import Path
import pandas as pd
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

    def insert_stock_info(self, data: pd.DataFrame):
        insert_data(self.db.get_connection(), "stocks", data)

    def insert_metadata(self, record: dict):
        insert_record(self.db.get_connection(), "metadata", record)

    def insert_market_data(self, data: pd.DataFrame):
        insert_data(self.db.get_connection(), "market", data)

    def insert_financial_data(self, data: pd.DataFrame):
        insert_data(self.db.get_connection(), "financials", data)

    def insert_insider_data(self, data: pd.DataFrame):
        insert_data(self.db.get_connection(), "insider", data)

    def insert_sp_data(self, data: pd.DataFrame):
        insert_data(self.db.get_connection(), "sp500", data)
        
    def delete_stock_data(self, tic: str):
        delete_data(self.db.get_connection(), "stocks", tic)

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
    ) -> pd.DataFrame | tuple:
        
        if tic:
            # fetch stock info
            rec = fetch_record(
                self.db.get_connection(),
                "stocks",
                {"tic": tic}
            )
            return rec
        else:
            # fetch all stocks info
            df = fetch_data(self.db.get_connection(), "stocks")
            df['last_update'] = pd.to_datetime(
                df['last_update'],
                format='ISO8601',
                errors='coerce'
            ).dt.date
            return df

    def fetch_metadata(
        self,
        tic: Optional[int] = None
    ) -> pd.DataFrame:
        
        if tic:
            # fetch metadata record of a single stock
            return fetch_record(self.db.get_connection(), "metadata", {"tic": tic})
        else:
            # fetch all metadata
            return fetch_data(self.db.get_connection(), "metadata")

    def fetch_market_data(
        self,
        tic: Optional[int] = None
    ) -> pd.DataFrame:
        
        if tic:
            # fetch single stock
            df = fetch_data(self.db.get_connection(), "market", {"tic": tic})
        else:
            # fetch all market records
            df = fetch_data(self.db.get_connection(), "market")
        
        # format dates
        df['date'] = pd.to_datetime(
            df['date'], format='ISO8601', errors='coerce'
        ).dt.date
        
        return df

    def fetch_financial_data(
        self,
        tic: Optional[int] = None
    ) -> pd.DataFrame:
        
        if tic:
            # fetch single stock
            df = fetch_data(self.db.get_connection(), "financials", {"tic": tic})
        else:
            # fetch all financials
            df = fetch_data(self.db.get_connection(), "financials")

        # format dates
        df['datadate'] = pd.to_datetime(
            df['datadate'], format='ISO8601', errors='coerce'
        ).dt.date
        
        df['rdq'] = pd.to_datetime(
            df['rdq'], format='ISO8601', errors='coerce'
        ).dt.date
        
        return df

    def fetch_insider_data(
        self,
        tic: Optional[int] = None
    ) -> pd.DataFrame:
        
        if tic:
            # fetch single stock
            df = fetch_data(self.db.get_connection(), "insider", {"tic": tic})
        else:
            # fetch all insider trading records
            df = fetch_data(self.db.get_connection(), "insider")

        # format dates
        df['trade_date'] = pd.to_datetime(
            df['trade_date'], format='ISO8601', errors='coerce'
        ).dt.date
        
        df['filling_date'] = pd.to_datetime(
            df['filling_date'], format='ISO8601', errors='coerce'
        ).dt.date

        return df

    def fetch_sp_data(self) -> pd.DataFrame:
        return fetch_data(self.db.get_connection(), "sp500")

    def close(self):
        self.db.close()
