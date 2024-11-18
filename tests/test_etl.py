import polars as pl

from stocksense.config import config
from stocksense.pipeline import ETL


def test_update_index_listings(mocker, mock_stock_data, mock_active_data):
    mock_db = mocker.Mock()
    mock_db.fetch_stock.return_value = mock_stock_data
    mock_db.update_stock.return_value = True
    mock_db.insert_stock.return_value = True
    mocker.patch("stocksense.pipeline.etl.DatabaseHandler", return_value=mock_db)

    # mock scraper
    mocker.patch(
        "stocksense.pipeline.scraper.Scraper.scrape_sp500_stock_info", return_value=mock_active_data
    )

    etl = ETL(config)
    etl.update_index_listings()

    # verify delisted stock was updated
    mock_db.update_stock.assert_any_call("OLD", {"spx_status": 0})

    # verify new stock was added
    mock_db.insert_stock.assert_called_once()
    insert_call_df = mock_db.insert_stock.call_args[0][0]
    assert "NEW" in insert_call_df["tic"].to_list()
    assert insert_call_df.filter(pl.col("tic") == "NEW")["spx_status"][0] == 1
    assert insert_call_df.filter(pl.col("tic") == "NEW")["active"][0] == 1
