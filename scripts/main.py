import datetime as dt
from loguru import logger
from pipeline import Etl
from database_handler import DatabaseHandler

def main():
    """
    Main function
    """

    logger.remove()
    logger.add(
        rf"./log/log_{dt.datetime.now().strftime('%Y%m%d')}.log",
        backtrace=False,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level} | "
            "{module}:{function}:{line} - {message}"
        ),
    )
    
    handler = Etl()
    #handler.ingest_historical_data()
    #handler.update_index_listings()
    handler.set_stocks()
    handler.extract()
    print('done.')


if __name__ == "__main__":
    main()
