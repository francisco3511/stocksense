import sqlite3
from sqlite3 import Error
from pathlib import Path
from loguru import logger


class DatabaseConnection:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.connection = None
        self._create_connection()

    def _create_connection(self) -> None:
        try:
            self.connection = sqlite3.connect(self.db_path)
        except Error as e:
            logger.error(f"Error connecting to database: {e}")

    def get_connection(self):
        return self.connection

    def close(self):
        if self.connection:
            self.connection.close()
