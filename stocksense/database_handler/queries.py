import polars as pl
from sqlite3 import Connection, Error
from loguru import logger
from typing import Optional


def insert_data(connection: Connection, table_name: str, data: pl.DataFrame) -> None:
    try:
        columns = ", ".join(data.columns)
        placeholders = ", ".join(["?"] * len(data.columns))
        sql = f"INSERT OR REPLACE INTO {table_name} ({columns}) VALUES ({placeholders})"
        cursor = connection.cursor()
        cursor.executemany(sql, data.to_numpy().tolist())
        connection.commit()
    except Error as e:
        logger.error(f"Error inserting data into {table_name}: {e}")


def insert_record(connection: Connection, table_name: str, record: dict) -> None:
    try:
        columns = ", ".join(record.keys())
        placeholders = ", ".join(["?"] * len(record))
        sql = f"INSERT OR REPLACE INTO {table_name} ({columns}) VALUES ({placeholders})"
        cursor = connection.cursor()
        cursor.execute(sql, list(record.values()))
        connection.commit()
    except Error as e:
        logger.error(f"Error inserting single record into {table_name}: {e}")


def update_data(
    connection: Connection, table_name: str, update_values: dict, condition: dict
) -> None:
    try:
        cursor = connection.cursor()
        set_clause = ", ".join([f"{col} = ?" for col in update_values.keys()])
        where_clause = " AND ".join([f"{col} = ?" for col in condition.keys()])
        sql = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"
        parameters = list(update_values.values()) + list(condition.values())
        cursor.execute(sql, parameters)
        connection.commit()
    except Error as e:
        logger.error(f"Error updating data in {table_name}: {e}")


def delete_data(connection: Connection, table_name: str, condition: dict) -> None:
    try:
        cursor = connection.cursor()
        where_clause = " AND ".join([f"{col} = ?" for col in condition.keys()])
        sql = f"DELETE FROM {table_name} WHERE {where_clause}"
        parameters = list(condition.values())
        cursor.execute(sql, parameters)
        connection.commit()
        logger.info(f"Data has been deleted from table {table_name}.")
    except Error as e:
        logger.error(f"Error deleting data from table {table_name}: {e}")


def delete_table(connection: Connection, table_name: str) -> bool:
    """
    Delete a table from the database.

    Parameters
    ----------
    connection : Connection
        Database connection object.
    table_name : str
        Name of table to delete.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    try:
        cursor = connection.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        connection.commit()
        logger.info(f"Successfully deleted table {table_name}")
        return True
    except Error as e:
        logger.error(f"Error deleting table {table_name}: {e}")
        return False


def count_data(connection: Connection, table_name: str, column: str) -> Optional[int]:
    try:
        sql = f"SELECT COUNT(DISTINCT {column}) FROM {table_name}"
        cursor = connection.cursor()
        cursor.execute(sql)
        count = cursor.fetchone()[0]
        return count
    except Error as e:
        logger.error(f"Error counting distinct values in column {column}: {e}")
        return None


def fetch_record(
    connection: Connection, table_name: str, condition: Optional[dict] = None
) -> Optional[pl.DataFrame]:
    try:
        cursor = connection.cursor()
        if condition:
            where_clause = " AND ".join([f"{col} = ?" for col in condition.keys()])
            sql = f"SELECT * FROM {table_name} WHERE {where_clause}"
            parameters = list(condition.values())
            cursor.execute(sql, parameters)
            row = cursor.fetchone()
        else:
            cursor.execute(f"SELECT * FROM {table_name}")
            row = cursor.fetchone()
        if row:
            columns = [description[0] for description in cursor.description]
            return pl.DataFrame(
                [row], schema=columns, orient="row", infer_schema_length=None
            )
        else:
            return None
    except Error as e:
        logger.error(f"Error fetching record from {table_name}: {e}")
        return None


def fetch_data(
    connection: Connection, table_name: str, condition: Optional[dict] = None
) -> Optional[pl.DataFrame]:
    try:
        cursor = connection.cursor()
        if condition:
            where_clause = " AND ".join([f"{col} = ?" for col in condition.keys()])
            sql = f"SELECT * FROM {table_name} WHERE {where_clause}"
            parameters = list(condition.values())
            cursor.execute(sql, parameters)
        else:
            cursor.execute(f"SELECT * FROM {table_name}")
        columns = [description[0] for description in cursor.description]
        data = cursor.fetchall()
        if data:
            return pl.DataFrame(
                data, schema=columns, orient="row", infer_schema_length=None
            )
        else:
            return None
    except Error as e:
        logger.error(f"Error fetching data from {table_name}: {e}")
        return None
