from sqlite3 import Connection, Error
from loguru import logger

def create_tables(connection: Connection) -> None:
    tables = {
        'stocks': '''
            CREATE TABLE IF NOT EXISTS stocks (
                tic TEXT PRIMARY KEY,
                name TEXT,
                sector TEXT,
                last_update TEXT,
                spx_status INTEGER,
                active INTEGER
            )
        ''',
        'metadata': '''
            CREATE TABLE IF NOT EXISTS metadata (
                tic TEXT PRIMARY KEY,
                shares_outstanding REAL,
                enterprise_value REAL,
                rec_key TEXT,
                forward_pe REAL
            )
        ''',
        'financials': '''
            CREATE TABLE IF NOT EXISTS financials (
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
        ''',
        'market': '''
            CREATE TABLE IF NOT EXISTS market (
                tic TEXT,
                date TEXT,
                close REAL,
                adj_close REAL,
                volume INTEGER,
                PRIMARY KEY (tic, date)
            )
        ''',
        'insider': '''
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
        ''',
        'sp500': '''
            CREATE TABLE IF NOT EXISTS sp500 (
                date TEXT PRIMARY KEY,
                close REAL,
                adj_close REAL,
                volume INTEGER
            )
        '''
    }

    try:
        cursor = connection.cursor()
        for table_name, table_schema in tables.items():
            cursor.execute(table_schema)
        connection.commit()
    except Error as e:
        logger.error(f"Error creating tables: {e}")
