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
@click.option("-f", "--force", is_flag=True, default=False, help="Force model retraining.")
@click.option(
    "-tdq",
    "--trade-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Trade date for model operations (format: YYYY-MM-DD)",
)
def main(update, train, score, force, trade_date):
    """
    CLI handling.
    """

    if update:
        etl_handler = ETL(config)
        etl_handler.update_index_listings()
        etl_handler.extract()

    if any([train, score]):
        data = prepare_data()
        stocks = DatabaseHandler().fetch_sp500_stocks()
        handler = ModelHandler(stocks, trade_date)
        if train:
            handler.train(data, retrain=force)
        if score:
            handler.score(data)


if __name__ == "__main__":
    main()
