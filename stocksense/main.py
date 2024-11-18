import click

from stocksense.config import config
from stocksense.model import ModelHandler
from stocksense.pipeline import ETL, clean, engineer_features


@click.command()
@click.option("-u", "--update", is_flag=True, help="Update stock data.")
@click.option("-t", "--train", is_flag=True, help="Train model.")
@click.option("-s", "--score", is_flag=True, help="Score stocks.")
def main(update, train, score):
    """
    CLI handling.
    """

    if update:
        etl_handler = ETL(config, stocks=["AAPL"])
        etl_handler.update_index_listings()
        etl_handler.extract()
    if train:
        data = engineer_features()
        data = clean(data)
        handler = ModelHandler()
        handler.train(data)
    if score:
        pass


if __name__ == "__main__":
    main()
