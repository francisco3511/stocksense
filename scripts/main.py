import click
import datetime as dt
from loguru import logger

from pipeline import ETL, Preprocess


@click.command("-u", "--update", is_flag=True, help="Extract data.")
@click.option("-p", "--preprocess", is_flag=True, help="Preprocess data.")
def main(update, preprocess):
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

    if update:
        # ETL process
        handler = ETL()
        # handler.ingest_all_historical_data()
        handler.extract()
    if preprocess:
        # data preprocessing
        proc = Preprocess()
        proc.process_data()
        today = dt.datetime.today().date()
        proc.save_data(f"proc_data_{today}")

    print('done.')


if __name__ == "__main__":
    main()
