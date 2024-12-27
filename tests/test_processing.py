import datetime as dt

import polars as pl
from polars.testing import assert_frame_equal

from stocksense.config import config
from stocksense.pipeline.preprocess import (
    clean,
    engineer_features,
    generate_quarter_dates,
    map_to_closest_split_factor,
)


def test_generate_quarter_dates():
    start_year = 2020
    end_year = 2021
    dates = generate_quarter_dates(start_year, end_year)

    assert len(dates) == 8  # 4 quarters * 2 years
    assert dates[0] == dt.datetime(2020, 3, 1)
    assert dates[-1] == dt.datetime(2021, 12, 1)


def test_map_to_closest_split_factor():
    assert map_to_closest_split_factor(0.49) == 0.5
    assert map_to_closest_split_factor(2.1) == 2.0
    assert map_to_closest_split_factor(3.9) == 4.0
    assert map_to_closest_split_factor(0.24) == 0.25


def test_feature_engineering(
    financial_data,
    market_data,
    insider_data,
    index_data,
    vix_data,
    info_data,
    processed_data,
    mocker,
):
    """Integration test for the full processing pipeline."""
    mock_db = mocker.Mock()
    mock_db.fetch_financial_data.return_value = financial_data
    mock_db.fetch_stock.return_value = info_data
    mock_db.fetch_market_data.return_value = market_data
    mock_db.fetch_insider_data.return_value = insider_data
    mock_db.fetch_index_data.return_value = index_data
    mock_db.fetch_vix_data.return_value = vix_data

    mocker.patch("stocksense.pipeline.preprocess.DatabaseHandler", return_value=mock_db)

    result_processed = engineer_features()

    # verify final output structure
    assert isinstance(result_processed, pl.DataFrame)
    assert not result_processed.is_empty()

    # check key computed features are present
    expected_features = [f for f in config.model.features if not f.startswith("sector_")]
    assert all(feature in result_processed.columns for feature in expected_features)

    # verify data equality after reprocessing
    assert_frame_equal(result_processed, processed_data)


def test_clean(
    processed_data,
    cleaned_data,
):
    # verify data equality after cleaning
    result_clean = clean(processed_data)
    assert_frame_equal(result_clean, cleaned_data)
