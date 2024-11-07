import click
import datetime as dt
from loguru import logger
from pathlib import Path

from config import get_config
from pipeline import ETL, Preprocess
from model import ModelHandler


@click.command()
@click.option("-u", "--update", is_flag=True, help="Update stock data.")
@click.option("-t", "--train", is_flag=True, help="Train model.")
@click.option("-s", "--score", is_flag=True, help="Score stocks.")
def main(update, train, score):
    """
    CLI handling.
    """

    log_path = Path("log/")
    log_path.mkdir(parents=True, exist_ok=True)
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
    if train:
        model_settings = get_config('model')
        data = Preprocess(
            features=model_settings['features'],
            targets=model_settings['targets']
        ).run()
        handler = ModelHandler()
        handler.train(data)
    if score:
        pass


if __name__ == "__main__":
    main()
