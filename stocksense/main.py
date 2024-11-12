import click
from model import ModelHandler
from pipeline import ETL, Preprocess


@click.command()
@click.option("-u", "--update", is_flag=True, help="Update stock data.")
@click.option("-t", "--train", is_flag=True, help="Train model.")
@click.option("-s", "--score", is_flag=True, help="Score stocks.")
def main(update, train, score):
    """
    CLI handling.
    """

    if update:
        etl_handler = ETL()
        etl_handler.update_index_listings()
        etl_handler.extract()
    if train:
        data = Preprocess().run()
        handler = ModelHandler()
        handler.train(data)
    if score:
        pass


if __name__ == "__main__":
    main()
