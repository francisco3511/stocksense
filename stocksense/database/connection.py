import sqlite3
from sqlite3 import Error

from loguru import logger

from stocksense.config import DATABASE_PATH


class DatabaseConnection:
    def __init__(self):
        self.db_path = DATABASE_PATH
        self.connection = None
        self._create_connection()

    def _create_connection(self) -> None:
        try:
            self.connection = sqlite3.connect(self.db_path)
        except Error as e:
            logger.error(f"Error connecting to database: {e}")

    def get_connection(self) -> sqlite3.Connection:
        if self.connection is None:
            raise ConnectionError("No database connection available.")
        return self.connection

    def close(self) -> None:
        if self.connection:
            self.connection.close()
