import click
import polars as pl

from stocksense.config import config
from stocksense.database_handler import DatabaseHandler
from stocksense.model import ModelHandler, PortfolioBuilder
from stocksense.pipeline import ETL, clean, engineer_features


def prepare_data() -> pl.DataFrame:
    """Prepare data for model operations."""
    data = engineer_features()
    return clean(data)


@click.command()
@click.option("-u", "--update", is_flag=True, help="Update stock data.")
@click.option("-t", "--train", is_flag=True, help="Train model.")
@click.option("-p", "--portfolio", is_flag=True, help="Build portfolio.")
@click.option("-f", "--force", is_flag=True, default=False, help="Force model retraining.")
@click.option(
    "-tdq",
    "--trade-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help=(
        "Trade date for model operations (format: YYYY-MM-DD)."
        "Must be the 1st of March, June, September or December."
    ),
)
def main(update, train, portfolio, force, trade_date):
    """
    CLI handling.
    """

    if update:
        etl_handler = ETL(config)
        etl_handler.extract()

    if any([train, portfolio]):
        data = prepare_data()
        constituents = DatabaseHandler().fetch_constituents(trade_date)
        handler = ModelHandler(trade_date)
        if train:
            handler.train(data, force)
        if portfolio:
            handler.score(data, constituents)
            portfolio = PortfolioBuilder(weighting="market_cap")
            portfolio.build_portfolio(trade_date)


if __name__ == "__main__":
    main()
