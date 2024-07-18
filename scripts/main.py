import datetime as dt
from loguru import logger

from pipeline import Etl, Preprocess


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

    #handler = Etl(["SW"])
    #handler.ingest_all_historical_data()
    #handler.update_index_listings()
    #handler.set_stocks()
    #handler.extract()
    
    proc = Preprocess()
    proc.process_data()
    
    today = dt.datetime.today().date()
    proc.save_data(f"proc_data_{today}")
    
    print('done.')


if __name__ == "__main__":
    main()
