import click
import datetime as dt
from loguru import logger

from pipeline import ETL, Preprocess
from model import ModelHandler


@click.command()
@click.option("-u", "--update", is_flag=True, help="Update stock data.")
@click.option("-p", "--preprocess", is_flag=True, help="Preprocess stock data.")
@click.option("-t", "--train_model", is_flag=True, help="Train model.")
def main(update, preprocess, train_model):
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
        etl_handler = ETL()
        etl_handler.update_index_listings()
        etl_handler.extract()
    if preprocess:
        proc = Preprocess()
        proc.run()
        proc.save_data()
    if train_model:
        model_handler = ModelHandler()
        model_handler.train()
        model_handler.score()


if __name__ == "__main__":
    main()
