from sqlite3 import Connection, Error

from loguru import logger


def create_tables(connection: Connection | None) -> None:
    tables = {
        "stocks": """
            CREATE TABLE IF NOT EXISTS stock (
                tic TEXT PRIMARY KEY,
                name TEXT,
                sector TEXT,
                date_added TEXT,
                date_removed TEXT
            )
        """,
        "info": """
            CREATE TABLE IF NOT EXISTS info (
                tic TEXT PRIMARY KEY,
                risk INT,
                beta REAL,
                volume REAL,
                trailing_pe REAL,
                forward_pe REAL,
                trailing_peg REAL,
                peg REAL,
                trailing_eps REAL,
                forward_eps REAL,
                price_book REAL,
                market_cap REAL,
                currency TEXT,
                shares_outstanding REAL,
                enterprise_value REAL,
                short_ratio REAL,
                curr_price REAL,
                fiftytwo_wc REAL,
                rec_key TEXT,
                target_low REAL,
                target_high REAL,
                target_mean REAL
            )
        """,
        "financial": """
            CREATE TABLE IF NOT EXISTS financial (
                tic TEXT,
                datadate TEXT,
                rdq TEXT NOT NULL,
                saleq REAL,
                cogsq REAL,
                xsgaq REAL,
                niq REAL,
                ebitdaq REAL,
                cshoq REAL,
                actq REAL,
                atq REAL,
                cheq REAL,
                rectq REAL,
                invtq REAL,
                ppentq REAL,
                lctq REAL,
                dlttq REAL,
                ltq REAL,
                req REAL,
                seqq REAL,
                oancfq REAL,
                ivncfq REAL,
                fincfq REAL,
                dvq REAL,
                capxq REAL,
                icaptq REAL,
                surprise_pct REAL,
                PRIMARY KEY (tic, datadate)
            )
        """,
        "market": """
            CREATE TABLE IF NOT EXISTS market (
                tic TEXT,
                date TEXT,
                close REAL,
                adj_close REAL,
                volume INTEGER,
                PRIMARY KEY (tic, date)
            )
        """,
        "insider": """
            CREATE TABLE IF NOT EXISTS insider (
                tic TEXT,
                filling_date TEXT,
                trade_date TEXT,
                owner_name TEXT,
                title TEXT,
                transaction_type TEXT,
                last_price TEXT,
                qty INTEGER,
                shares_held TEXT,
                owned TEXT,
                value TEXT,
                PRIMARY KEY (tic, filling_date, owner_name, transaction_type, value)
            )
        """,
        "sp500": """
            CREATE TABLE IF NOT EXISTS sp500 (
                date TEXT PRIMARY KEY,
                close REAL,
                volume INTEGER
            )
        """,
        "vix": """
            CREATE TABLE IF NOT EXISTS vix (
                date TEXT PRIMARY KEY,
                close REAL
            )
        """,
    }

    try:
        cursor = connection.cursor()
        for table_schema in tables.values():
            cursor.execute(table_schema)
        connection.commit()
        logger.success("Tables created successfully")
    except Error as e:
        logger.error(f"Error creating tables: {e}")
