import click
import datetime as dt
from loguru import logger

from pipeline import ETL, Preprocess


@click.command()
@click.option("-u", "--update", is_flag=True, help="Update stock data.")
@click.option("-p", "--preprocess", is_flag=True, help="Preprocess stock data.")
@click.option("-t", "--train", is_flag=True, help="Train model.")
def main(update, preprocess, train):
    """
    CLI handling.
    """

    logger.remove()
    logger.add(
        rf"log/log_{dt.datetime.now().strftime('%Y%m%d')}.log",
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
        handler.extract()
    if preprocess:
        # data preprocessing
        proc = Preprocess()
        proc.run()
        proc.save_data()
    if train:
        pass
    print('done.')


if __name__ == "__main__":
    main()
