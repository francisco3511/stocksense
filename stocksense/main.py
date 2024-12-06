from datetime import datetime
from typing import Optional

import click
import polars as pl

from stocksense import __version__
from stocksense.config import config
from stocksense.database_handler import DatabaseHandler
from stocksense.model import ModelHandler, PortfolioBuilder
from stocksense.pipeline import ETL, clean, engineer_features


def validate_trade_date(ctx, param, value: Optional[datetime]) -> Optional[datetime]:
    """Validate that trade date is 1st of Mar/Jun/Sep/Dec."""
    if value is None:
        return value
    valid_months = {3, 6, 9, 12}
    if value.day != 1 or value.month not in valid_months:
        raise click.BadParameter("Trade date must be the 1st of March, June, September or December")
    return value


def prepare_data() -> pl.DataFrame:
    """Prepare data for model operations."""
    data = engineer_features()
    data = clean(data)
    return data


@click.group()
@click.version_option(version=__version__, prog_name="stocksense")
def cli():
    """Stocksense CLI - Stock analytics and portfolio management tool."""
    pass


@cli.command()
@click.option(
    "-c", "--config-path", type=click.Path(exists=True), help="Path to custom configuration file"
)
def update():
    """
    Update stock database with latest market data.
    """
    click.echo("Updating stock database...")
    etl_handler = ETL(config)
    etl_handler.extract()
    click.echo("Database update complete.")


@cli.command()
@click.option(
    "-tdq",
    "--trade-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    callback=validate_trade_date,
    required=True,
    help="Trade date (YYYY-MM-DD). Must be 1st of Mar/Jun/Sep/Dec.",
)
@click.option("-f", "--force", is_flag=True, help="Force model retraining even if model exists.")
def train(trade_date: datetime, force: bool):
    """Train the prediction model for a specific trade date."""
    click.echo(f"Training model for trade date: {trade_date.date()}")
    data = prepare_data()
    handler = ModelHandler(trade_date)
    handler.train(data, force)
    click.echo("Model training complete!")


@cli.command()
@click.option(
    "-tdq",
    "--trade-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    callback=validate_trade_date,
    required=True,
    help="Trade date (YYYY-MM-DD). Must be 1st of Mar/Jun/Sep/Dec.",
)
@click.option(
    "-w",
    "--weighting",
    type=click.Choice(["market_cap", "equal"], case_sensitive=False),
    default="market_cap",
    help="Portfolio weighting strategy.",
)
def portfolio(trade_date: datetime, weighting: str):
    """Build investment portfolio for a specific trade date."""
    click.echo(f"Building portfolio for trade date: {trade_date.date()}")

    data = prepare_data()
    constituents = DatabaseHandler().fetch_constituents(trade_date)

    handler = ModelHandler(trade_date)
    handler.score(data, constituents)

    portfolio = PortfolioBuilder(weighting=weighting)
    portfolio.build_portfolio(trade_date)

    click.echo("Portfolio construction complete!")


def main():
    """CLI entry point."""
    cli(prog_name="stocksense")


if __name__ == "__main__":
    main()
