from datetime import datetime

import click

from stocksense.config import config
from stocksense.database_handler import DatabaseHandler
from stocksense.model import ModelHandler
from stocksense.pipeline import ETL, clean, engineer_features


def prepare_data():
    """Prepare data for model operations."""
    data = engineer_features()
    return clean(data)


@click.command()
@click.option("-u", "--update", is_flag=True, help="Update stock data.")
@click.option("-t", "--train", is_flag=True, help="Train model.")
@click.option("-s", "--score", is_flag=True, help="Score stocks.")
@click.option("-b", "--backtest", is_flag=True, help="Backtest model.")
def main(update, train, score, backtest):
    """
    CLI handling.
    """

    if update:
        etl_handler = ETL(config)
        etl_handler.update_index_listings()
        etl_handler.extract()

    if any([train, score, backtest]):
        data = prepare_data()
        stocks = DatabaseHandler().fetch_sp500_stocks()
        handler = ModelHandler(stocks)
        if train:
            handler.train(data)
        if score:
            handler.score(data)
        if backtest:
            handler.backtest(data, [datetime(2023, 9, 1)])


if __name__ == "__main__":
    main()
