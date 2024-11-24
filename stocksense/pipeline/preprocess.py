import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import polars_talib as plta
from loguru import logger

from stocksense.config import config
from stocksense.database_handler import DatabaseHandler

DATA_PATH = Path(__file__).parents[1] / "data"
FIXTURE_PATH = Path(__file__).parents[2] / "tests" / "fixtures"


def engineer_features() -> pl.DataFrame:
    """
    Runs main data processing pipeline.
    """
    logger.info("START processing stock data")
    db = DatabaseHandler()

    try:
        # fetch all required data
        df = db.fetch_financial_data()
        info = db.fetch_stock()
        market_df = db.fetch_market_data().sort(["tic", "date"])
        insider_df = db.fetch_insider_data()
        index_data = db.fetch_index_data()
        vix_data = db.fetch_vix_data()

        # feature engineering
        logger.info("START feature engineering")
        df = compute_trade_date(df)
        df = adjust_shares(df)
        df = compute_insider_trading_features(df, insider_df)
        df = compute_financial_features(df)
        df = compute_sp500_features(df, index_data)
        df = compute_vix_features(df, vix_data)
        df = compute_market_features(df, market_df, index_data)
        df = compute_growth_features(df)
        df = compute_piotroski_score(df)
        df = compute_performance_targets(df)
        df = compute_sector_dummies(df, info)

        logger.success(f"END {df.shape[0]} rows PROCESSED")
    except Exception as e:
        logger.error(f"FAILED processing stock data: {e}")
        raise e
    return df


def clean(df: pl.DataFrame) -> pl.DataFrame:
    """
    Clean the data by removing rows with missing values.

    Parameters
    ----------
    data : pl.DataFrame
        Financial features dataset.

    Returns
    -------
    pl.DataFrame
        Filtered and processed data.
    """

    logger.info("START cleaning data")

    df = df.filter(pl.col("tdq") <= pl.lit(dt.datetime.today().date()))
    growth_alias = ["mom", "sos", "qoq", "yoy", "2y", "return"]
    growth_vars = [f for f in df.columns if any(xf in f for xf in growth_alias)]

    for feature in [f for f in df.columns if any(xf in f for xf in growth_vars)]:
        df = df.with_columns(pl.col(feature) * 100)
        df = df.with_columns(df.with_columns(pl.col(feature).clip(-2000, 2000)))

    float_cols = df.select(pl.col(pl.Float64)).columns
    df = df.with_columns(
        [
            pl.col(col)
            .replace(float("inf"), float("nan"))
            .replace(float("-inf"), float("nan"))
            .alias(col)
            for col in float_cols
        ]
    )
    df = df.filter(~pl.all_horizontal(pl.col("niq_2y").is_null()))
    df = df.sort(["tic", "tdq"]).unique(subset=["tic", "tdq"], keep="last", maintain_order=True)
    df = df.sort(["tic", "rdq"])

    logger.success(f"{df.shape[0]} rows retained after CLEANING")
    return df


def compute_trade_date(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute trade date intervals, to be used as a proxy for quarters.
    These represent the financial observations in which models will be trained on.

    Parameters
    ----------
    df : pl.DataFrame
        Financial data of a given stock.

    Returns
    -------
    pl.DataFrame
        Data with trade date intervals.
    """

    # correct rdq if it is the same as quarter end date
    df = df.with_columns(
        pl.when(pl.col("rdq") == pl.col("datadate"))
        .then(pl.col("datadate") + pl.duration(days=90))
        .otherwise(pl.col("rdq"))
        .alias("rdq")
    )

    min_year = df["rdq"].dt.year().min()
    max_year = df["rdq"].dt.year().max()

    quarter_dates = generate_quarter_dates(min_year, max_year)
    quarter_df = pl.DataFrame({"tdq": quarter_dates}).with_columns(pl.col("tdq").dt.date())

    df = df.sort(by=["rdq", "tic"])
    df = df.join_asof(quarter_df, left_on="rdq", right_on="tdq", strategy="forward")
    return df.sort(by=["tic", "rdq"])


def generate_quarter_dates(start_year: int, end_year: int) -> list:
    """
    Function to generate quarter-end dates for a range of years.
    """
    quarter_end_dates = []
    for year in range(start_year, end_year + 1):
        quarter_end_dates.extend(
            [
                dt.datetime(year, 3, 1),
                dt.datetime(year, 6, 1),
                dt.datetime(year, 9, 1),
                dt.datetime(year, 12, 1),
            ]
        )
    return quarter_end_dates


def map_to_closest_split_factor(approx_factor: float) -> float:
    common_split_ratios = np.array([1, 0.5, 0.33, 0.25, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30])
    diffs = np.abs(common_split_ratios - approx_factor)
    closest_index = np.argmin(diffs)
    return common_split_ratios[closest_index]


def adjust_shares(df: pl.DataFrame) -> pl.DataFrame:
    """
    Computes adjusted common outstanding shares,
    for historical analysis, based on stock-split detection.

    Parameters
    ----------
    df : pl.DataFrame
        Financial data of a given stock.

    Returns
    -------
    pl.DataFrame
        Data with adjusted outstanding shares.
    """
    df = df.with_columns(pl.col("cshoq").fill_null(strategy="backward").over("tic"))
    df = df.with_columns(
        (pl.col("cshoq") / pl.col("cshoq").shift()).over("tic").alias("csho_ratio")
    )
    df = df.with_columns(
        ((pl.col("cshoq") / pl.col("cshoq").shift() - 1) > 0.25).over("tic").alias("stock_split")
    )
    df = df.with_columns(pl.col("stock_split").fill_null(False))
    df = df.with_columns(
        pl.col("csho_ratio")
        .map_elements(map_to_closest_split_factor, return_dtype=pl.Float64)
        .alias("split_factor")
    )

    # compute the adjustment factor: 1 if no split occurred, otherwise the split factor
    df = df.with_columns(
        pl.when(pl.col("stock_split"))
        .then(pl.col("split_factor"))
        .otherwise(1.0)
        .shift(-1)
        .fill_null(strategy="forward")
        .alias("adjustment_factor")
    )

    # compute cumulative product of adjustment factors in reverse
    df = df.sort(by=["tic", "datadate"]).with_columns(
        pl.col("adjustment_factor")
        .cum_prod(reverse=True)
        .over("tic")
        .alias("cum_adjustment_factor")
    )

    # apply the cumulative adjustment to the financial data
    df = df.with_columns((pl.col("cshoq") * pl.col("cum_adjustment_factor")).alias("cshoq"))
    df = df.with_columns(pl.col("stock_split").cast(pl.Int8))
    df = df.sort(by=["tic", "rdq"])
    return df.drop(["csho_ratio", "split_factor", "adjustment_factor", "cum_adjustment_factor"])


def compute_insider_purchases(df: pl.DataFrame, insider_df: pl.DataFrame) -> pl.DataFrame:
    """
    Computes insider purchases quarterly features.
    """

    insider_filtered_lazy = insider_df.lazy().filter((pl.col("transaction_type") == "P - Purchase"))
    df_lazy = df.lazy()

    df_filtered_lazy = df_lazy.join(
        insider_filtered_lazy, how="inner", left_on=["tic"], right_on=["tic"]
    ).filter(
        (pl.col("filling_date") < pl.col("tdq"))
        & (pl.col("filling_date") >= pl.col("tdq").dt.offset_by("-3mo"))
    )
    df_agg_lazy = df_filtered_lazy.group_by(["tic", "tdq"]).agg(
        [
            pl.col("filling_date").count().alias("n_purch"),
            (
                pl.col("value")
                .str.replace_all(r"[\$,€]", "")
                .str.replace_all(",", "")
                .cast(pl.Float64)
                / 1000000
            )
            .sum()
            .round(3)
            .alias("val_purch"),
        ]
    )
    result = df_lazy.join(df_agg_lazy, on=["tic", "tdq"], how="left").fill_null(0)
    return result.collect()


def compute_insider_sales(df: pl.DataFrame, insider_df: pl.DataFrame) -> pl.DataFrame:
    """
    Computes insider sales quarterly features.
    """

    insider_filtered_lazy = insider_df.lazy().filter(
        (pl.col("transaction_type") == "S - Sale") | (pl.col("transaction_type") == "S - Sale+OE")
    )
    df_lazy = df.lazy()

    df_filtered_lazy = df_lazy.join(
        insider_filtered_lazy, how="inner", left_on=["tic"], right_on=["tic"]
    ).filter(
        (pl.col("filling_date") < pl.col("tdq"))
        & (pl.col("filling_date") >= pl.col("tdq").dt.offset_by("-3mo"))
    )

    df_agg_lazy = df_filtered_lazy.group_by(["tic", "tdq"]).agg(
        [
            pl.col("filling_date").count().alias("n_sales"),
            (
                -pl.col("value")
                .str.replace_all(r"[\$,€]", "")
                .str.replace_all(",", "")
                .cast(pl.Float64)
                / 1000000
            )
            .sum()
            .round(3)
            .alias("val_sales"),
        ]
    )

    result = df_lazy.join(df_agg_lazy, on=["tic", "tdq"], how="left").fill_null(0)
    return result.collect()


def compute_insider_trading_features(df: pl.DataFrame, insider_df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute insider trading features, i.e. insider
    sales, purchases and balances.

    Parameters
    ----------
    df : pl.DataFrame
        Main dataset.
    insider_df : pl.DataFrame
        Insider trading records.

    Returns
    -------
    pl.DataFrame
        Main dataset w/ insider trading features
    """

    df = compute_insider_purchases(df, insider_df)
    df = compute_insider_sales(df, insider_df)
    df = df.with_columns((pl.col("val_sales") - pl.col("val_purch")).alias("insider_balance"))
    return df


def compute_financial_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Computes a selected number of financial ratios.

    Parameters
    ----------
    df : pl.DataFrame
        Financial data of a given stock.

    Returns
    -------
    pl.DataFrame
        Data with additional columns.
    """
    df = df.with_columns(
        pl.when(pl.col("niq") == 0).then(pl.lit(None)).otherwise(pl.col("niq")).alias("niq")
    )
    return (
        df.lazy()
        .with_columns(
            (pl.col("niq").rolling_sum(4) / pl.col("atq")).over("tic").alias("roa"),
            (pl.col("niq") / pl.col("icaptq")).over("tic").alias("roi"),
            (pl.col("niq").rolling_sum(4) / pl.col("seqq")).over("tic").alias("roe"),
            ((pl.col("saleq") - pl.col("cogsq")) / pl.col("saleq")).alias("gpm"),
            (pl.col("ebitdaq") / pl.col("saleq")).alias("ebitdam"),
            (pl.col("oancfq") / pl.col("saleq")).alias("cfm"),
            (pl.col("oancfq") - pl.col("capxq")).alias("fcf"),
            (pl.col("actq") / pl.col("lctq")).alias("cr"),
            ((pl.col("rectq") + pl.col("cheq")) / pl.col("lctq")).alias("qr"),
            (pl.col("cheq") / pl.col("lctq")).alias("csr"),
            (pl.col("ltq") / pl.col("atq")).alias("dr"),
            (pl.col("ltq") / pl.col("seqq")).alias("der"),
            (pl.col("ltq") / pl.col("ebitdaq")).alias("debitda"),
            (pl.col("dlttq") / pl.col("atq")).alias("ltda"),
            ((pl.col("oancfq") - pl.col("capxq")) / pl.col("dlttq")).alias("ltcr"),
            (pl.col("saleq") / pl.col("invtq").rolling_mean(2)).over("tic").alias("itr"),
            (pl.col("saleq") / pl.col("rectq").rolling_mean(2)).over("tic").alias("rtr"),
            (pl.col("saleq") / pl.col("atq").rolling_mean(2)).over("tic").alias("atr"),
            pl.col("atq").log().alias("size"),
        )
        .collect()
    )


def compute_sp500_features(df: pl.DataFrame, index_df: pl.DataFrame) -> pl.DataFrame:
    """
    Process S&P500 index price data.

    Returns
    -------
    pl.DataFrame
        Processed data, incl. forward and past return rates.
    """

    index_df = index_df.sort(by=["date"])
    df = df.sort(by=["tdq", "tic"])

    # compute index past returns
    index_df = index_df.with_columns(
        pl.col("close").pct_change(config.processing.trading_days_month).alias("index_mom"),
        pl.col("close").pct_change(config.processing.trading_days_quarter).alias("index_qoq"),
        pl.col("close").pct_change(config.processing.trading_days_semester).alias("index_sos"),
        pl.col("close").pct_change(config.processing.trading_days_year).alias("index_yoy"),
        pl.col("close").pct_change(config.processing.trading_days_2year).alias("index_2y"),
    )

    # compute volatily of index
    index_df = index_df.with_columns(
        (pl.col("close") / pl.col("close").shift(1)).log().alias("log_return")
    ).with_columns(
        pl.col("log_return")
        .rolling_std(config.processing.trading_days_month)
        .alias("index_vol_mom"),
        pl.col("log_return")
        .rolling_std(config.processing.trading_days_quarter)
        .alias("index_vol_qoq"),
        pl.col("log_return")
        .rolling_std(config.processing.trading_days_semester)
        .alias("index_vol_sos"),
        pl.col("log_return")
        .rolling_std(config.processing.trading_days_year)
        .alias("index_vol_yoy"),
        pl.col("log_return")
        .rolling_std(config.processing.trading_days_2year)
        .alias("index_vol_2y"),
    )

    index_df = index_df.rename(
        {
            "date": "index_date",
            "close": "index_close",
        }
    )

    index_df = index_df.select(
        [
            "index_date",
            "index_close",
            "index_mom",
            "index_qoq",
            "index_sos",
            "index_yoy",
            "index_2y",
            "index_vol_mom",
            "index_vol_qoq",
            "index_vol_sos",
            "index_vol_yoy",
            "index_vol_2y",
        ]
    )
    df = df.join_asof(
        index_df,
        left_on="tdq",
        right_on="index_date",
        strategy="backward",
        tolerance=dt.timedelta(days=7),
    )
    return df.sort(by=["tic", "rdq"])


def compute_vix_features(df: pl.DataFrame, vix_df: pl.DataFrame) -> pl.DataFrame:
    """
    Process VIX data.
    """

    vix_df = vix_df.sort(by=["date"])
    df = df.sort(by=["tdq", "tic"])

    vix_df = vix_df.with_columns(
        pl.col("close").alias("market_fear"),
        pl.col("close").rolling_mean(30).alias("fear_ma30"),
        (pl.col("close") > pl.col("close").rolling_max(252) * 0.8).cast(pl.Int8).alias("high_fear"),
        (pl.col("close") < pl.col("close").rolling_min(252) * 1.2).cast(pl.Int8).alias("low_fear"),
    )

    vix_df = vix_df.rename({"date": "vix_date"})
    vix_df = vix_df.select(
        [
            "vix_date",
            "market_fear",
            "fear_ma30",
            "high_fear",
            "low_fear",
        ]
    )
    df = df.join_asof(
        vix_df,
        left_on="tdq",
        right_on="vix_date",
        strategy="backward",
        tolerance=dt.timedelta(days=7),
    )
    return df.sort(by=["tic", "rdq"])


def compute_market_features(
    df: pl.DataFrame, market_df: pl.DataFrame, index_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Compute market-related ratios.

    Parameters
    ----------
    df : pl.DataFrame
        Financial data of a given stock.
    market_df : pl.DataFrame
        Market data.
    index_df : pl.DataFrame
        Index price data.

    Returns
    -------
    pl.DataFrame
        Main dataset with added ratios.
    """

    market_df = compute_volume_features(market_df)
    market_df = compute_daily_momentum_features(market_df)
    market_df = compute_daily_volatility_features(market_df)

    df = df.sort(by=["rdq", "tic"])
    df = df.join_asof(
        market_df.drop(["volume"]),
        left_on="tdq",
        right_on="date",
        by="tic",
        strategy="backward",
        tolerance=dt.timedelta(days=7),
    ).join_asof(
        market_df.select(["date", "tic", "close"]).rename(
            {"date": "rdq_date", "close": "rdq_close"}
        ),
        left_on="rdq",
        right_on="rdq_date",
        by="tic",
        strategy="forward",
        tolerance=dt.timedelta(days=7),
    )
    df = df.sort(by=["tic", "rdq"])

    df = compute_hybrid_features(df)
    return df


def compute_volume_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute normalized volume features.

    Parameters
    ----------
    df : pl.DataFrame
        Market data with volume information

    Returns
    -------
    pl.DataFrame
        DataFrame with normalized volume features
    """
    df = df.with_columns(
        [
            pl.col("volume").rolling_mean(20).over("tic").alias("volume_ma20_raw"),
            pl.col("volume").rolling_mean(50).over("tic").alias("volume_ma50_raw"),
            pl.col("volume").rolling_mean(252).over("tic").alias("volume_annual_mean"),
        ]
    )

    return df.with_columns(
        [
            (pl.col("volume_ma20_raw") / pl.col("volume_annual_mean") * 100).alias("volume_ma20"),
            (pl.col("volume_ma50_raw") / pl.col("volume_annual_mean") * 100).alias("volume_ma50"),
            (pl.col("volume") / pl.col("volume_annual_mean") * 100).alias("volume_ratio"),
        ]
    ).drop(["volume_ma20_raw", "volume_ma50_raw", "volume_annual_mean"])


def compute_daily_momentum_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute daily price momentum features.

    Parameters
    ----------
    df : pl.DataFrame
        Daily market data for all stocks.

    Returns
    -------
    pl.DataFrame
        Market data with momementum features.
    """
    return df.with_columns(
        plta.rsi(pl.col("close"), timeperiod=14).over("tic").alias("rsi_14d"),
        plta.rsi(pl.col("close"), timeperiod=30).over("tic").alias("rsi_30d"),
        plta.rsi(pl.col("close"), timeperiod=60).over("tic").alias("rsi_60d"),
        plta.rsi(pl.col("close"), timeperiod=90).over("tic").alias("rsi_90d"),
        plta.rsi(pl.col("close"), timeperiod=360).over("tic").alias("rsi_1y"),
        pl.col("close")
        .pct_change(config.processing.trading_days_month)
        .over("tic")
        .alias("price_mom"),
        pl.col("close")
        .pct_change(config.processing.trading_days_quarter)
        .over("tic")
        .alias("price_qoq"),
        pl.col("close")
        .pct_change(config.processing.trading_days_year)
        .over("tic")
        .alias("price_yoy"),
        pl.col("close")
        .pct_change(config.processing.trading_days_2year)
        .over("tic")
        .alias("price_2y"),
    )


def compute_daily_volatility_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute daily price volatility features.

    Parameters
    ----------
    df : pl.DataFrame
        Daily market data for all stocks.

    Returns
    -------
    pl.DataFrame
        Market data with momementum features.
    """
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1)).log().over("tic").alias("log_return")
    )
    return df.with_columns(
        pl.col("log_return")
        .rolling_std(config.processing.trading_days_month)
        .over("tic")
        .alias("vol_mom"),
        pl.col("log_return")
        .rolling_std(config.processing.trading_days_quarter)
        .over("tic")
        .alias("vol_qoq"),
        pl.col("log_return")
        .rolling_std(config.processing.trading_days_year)
        .over("tic")
        .alias("vol_yoy"),
    )


def compute_hybrid_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute market and financial ratios.

    Parameters
    ----------
    df : pl.DataFrame
        Main dataset.

    Returns
    -------
    pl.DataFrame
        Dataset with market/financial ratios.
    """
    df = df.sort(by=["tic", "rdq"])
    return (
        df.with_columns(
            ((pl.col("close") - pl.col("rdq_close")) / pl.col("rdq_close") * 100).alias(
                "earn_drift"
            ),
            (pl.col("price_mom") / pl.col("index_mom")).alias("momentum_mom"),
            (pl.col("price_qoq") / pl.col("index_qoq")).alias("momentum_qoq"),
            (pl.col("price_yoy") / pl.col("index_yoy")).alias("momentum_yoy"),
            (pl.col("price_2y") / pl.col("index_2y")).alias("momentum_2y"),
            (pl.col("vol_mom") / pl.col("index_vol_mom")).alias("rel_vol_mom"),
            (pl.col("vol_qoq") / pl.col("index_vol_qoq")).alias("rel_vol_qoq"),
            (pl.col("vol_yoy") / pl.col("index_vol_yoy")).alias("rel_vol_yoy"),
            (pl.col("niq").rolling_sum(4) / pl.col("cshoq")).over("tic").alias("eps"),
        )
        .with_columns(
            (pl.col("close") / pl.col("eps")).alias("pe"),
            (pl.col("close") * pl.col("cshoq")).alias("mkt_cap"),
        )
        .with_columns(
            (pl.col("mkt_cap") + pl.col("ltq") - pl.col("cheq")).alias("ev"),
            (pl.col("mkt_cap") / (pl.col("atq") - pl.col("ltq"))).alias("pb"),
            (pl.col("mkt_cap") / pl.col("saleq").rolling_sum(4)).over("tic").alias("ps"),
        )
        .with_columns(
            (pl.col("ev") / pl.col("ebitdaq").rolling_sum(4)).over("tic").alias("ev_ebitda")
        )
    )


def _compute_growth_rate(col: str, periods: int, suffix: str) -> pl.Expr:
    """
    Compute growth rate between lagged value and current value.

    Parameters
    ----------
    col : str
        Column name to compute growth for
    periods : int
        Number of periods to shift
    suffix : str
        Suffix for the output column name

    Returns
    -------
    pl.Expr
        Polars expression for the growth calculation
    """
    return (
        ((pl.col(col) - pl.col(col).shift(periods)) / pl.col(col).shift(periods).abs())
        .over("tic")
        .alias(f"{col}_{suffix}")
    )


def compute_growth_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute rolling growth statistics for financial metrics.

    Parameters
    ----------
    df : pl.DataFrame
        Financial data of a given stock.

    Returns
    -------
    pl.DataFrame
        Data with additional growth ratio columns.
    """

    quarter_lag = 1
    year_lag = 4
    two_year_lag = 8

    metrics = {
        "niq": [quarter_lag, year_lag, two_year_lag],
        "saleq": [year_lag, two_year_lag],
        "ltq": [quarter_lag, year_lag, two_year_lag],
        "dlttq": [year_lag, two_year_lag],
        "gpm": [year_lag, two_year_lag],
        "roa": [year_lag, two_year_lag],
        "roi": [year_lag, two_year_lag],
        "roe": [year_lag, two_year_lag],
        "fcf": [year_lag, two_year_lag],
        "cr": [year_lag, two_year_lag],
        "qr": [year_lag, two_year_lag],
        "der": [year_lag, two_year_lag],
        "dr": [year_lag, two_year_lag],
        "ltda": [year_lag, two_year_lag],
        "pe": [year_lag, two_year_lag],
        "pb": [year_lag, two_year_lag],
        "ps": [year_lag, two_year_lag],
        "eps": [year_lag, two_year_lag],
        "ev_ebitda": [year_lag, two_year_lag],
        "ltcr": [year_lag],
        "itr": [year_lag],
        "rtr": [year_lag],
        "atr": [year_lag],
    }

    expressions = []

    # add standard growth calculations
    df = df.sort(by=["tic", "rdq"])
    for metric, periods in metrics.items():
        for period in periods:
            suffix = "qoq" if period == quarter_lag else "yoy" if period == year_lag else "2y"
            expressions.append(_compute_growth_rate(metric, period, suffix))

    return df.lazy().with_columns(expressions).collect()


def compute_piotroski_score(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute Piotroski F-Score components (9 points total):
    1. ROA > 0 (1 point)
    2. Operating Cash Flow > 0 (1 point)
    3. ROA increasing (1 point)
    4. OCF > ROA (1 point)
    5. Decrease in leverage (LTD) (1 point)
    6. Increase in current ratio (1 point)
    7. No new shares issued (1 point)
    8. Increase in gross margin (1 point)
    9. Increase in asset turnover (1 point)
    """
    df = df.with_columns(
        [
            # Profitability Signals (4 points)
            (pl.col("roa") > 0).cast(pl.Int8).alias("f_roa"),
            (pl.col("oancfq") > 0).cast(pl.Int8).alias("f_ocf"),
            (pl.col("roa") > pl.col("roa").shift(4)).over("tic").cast(pl.Int8).alias("f_droa"),
            (pl.col("oancfq").rolling_sum(4) / pl.col("atq") > pl.col("roa"))
            .over("tic")
            .cast(pl.Int8)
            .alias("f_accrual"),
            # Leverage, Liquidity, and Source of Funds (3 points)
            (pl.col("ltq") < pl.col("ltq").shift(4)).over("tic").cast(pl.Int8).alias("f_dlever"),
            (pl.col("cr") > pl.col("cr").shift(4)).over("tic").cast(pl.Int8).alias("f_dliquid"),
            (pl.col("cshoq") <= pl.col("cshoq").shift(4)).over("tic").cast(pl.Int8).alias("f_dshr"),
            # Operating Efficiency (2 points)
            (pl.col("gpm") > pl.col("gpm").shift(4)).over("tic").cast(pl.Int8).alias("f_dgm"),
            (pl.col("atr") > pl.col("atr").shift(4)).over("tic").cast(pl.Int8).alias("f_dturn"),
        ]
    ).with_columns(
        [
            # Total F-Score (sum of all components)
            (
                pl.col("f_roa")
                + pl.col("f_ocf")
                + pl.col("f_droa")
                + pl.col("f_accrual")
                + pl.col("f_dlever")
                + pl.col("f_dliquid")
                + pl.col("f_dshr")
                + pl.col("f_dgm")
                + pl.col("f_dturn")
            ).alias("f_score")
        ]
    )
    df = df.with_columns(
        (pl.col("f_score") - pl.col("f_score").shift(1)).over("tic").alias("f_score_gr1"),
        (pl.col("f_score") - pl.col("f_score").shift(4)).over("tic").alias("f_score_gr4"),
    )
    component_cols = [
        "f_roa",
        "f_ocf",
        "f_droa",
        "f_accrual",
        "f_dlever",
        "f_dliquid",
        "f_dshr",
        "f_dgm",
        "f_dturn",
    ]
    return df.drop(component_cols)


def compute_performance_targets(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute target forward performance ratios.

    Parameters
    ----------
    df : pl.DataFrame
        Main dataset.

    Returns
    -------
    pl.DataFrame
        Dataset with each observation associated to forward returns and flags.
    """
    df = df.sort(["tic", "rdq"])
    df = df.with_columns(
        (
            (
                pl.col("index_close").shift(-config.processing.prediction_horizon)
                / pl.col("index_close")
            )
            - 1
        )
        .over("tic")
        .alias("index_freturn"),
        (
            (pl.col("adj_close").shift(-config.processing.prediction_horizon) / pl.col("adj_close"))
            - 1
        )
        .over("tic")
        .alias("freturn"),
        ((pl.col("adj_close").shift(-1) / pl.col("adj_close")) - 1).over("tic").alias("freturn_1q"),
        ((pl.col("adj_close").shift(-2) / pl.col("adj_close")) - 1).over("tic").alias("freturn_2q"),
        ((pl.col("adj_close").shift(-3) / pl.col("adj_close")) - 1).over("tic").alias("freturn_3q"),
        ((pl.col("adj_close").shift(-4) / pl.col("adj_close")) - 1).over("tic").alias("freturn_4q"),
    )
    df = df.with_columns((pl.col("freturn") - pl.col("index_freturn")).alias("adj_freturn"))
    df = df.with_columns(
        (pl.col("adj_freturn") > config.processing.over_performance_threshold)
        .cast(pl.Int8)
        .alias("adj_fperf"),
        (pl.col("freturn_1q") > config.processing.performance_threshold)
        .cast(pl.Int8)
        .alias("fperf_1q"),
        (pl.col("freturn_2q") > config.processing.performance_threshold)
        .cast(pl.Int8)
        .alias("fperf_2q"),
        (pl.col("freturn_3q") > config.processing.performance_threshold)
        .cast(pl.Int8)
        .alias("fperf_3q"),
        (pl.col("freturn_4q") > config.processing.performance_threshold)
        .cast(pl.Int8)
        .alias("fperf_4q"),
    ).with_columns(
        ((pl.col("fperf_1q") + pl.col("fperf_2q") + pl.col("fperf_3q") + pl.col("fperf_4q")) > 0)
        .cast(pl.Int8)
        .alias("fperf")
    )
    component_cols = [
        "freturn_1q",
        "freturn_2q",
        "freturn_3q",
        "freturn_4q",
        "fperf_1q",
        "fperf_2q",
        "fperf_3q",
        "fperf_4q",
    ]
    return df.drop(component_cols)


def compute_sector_dummies(df: pl.DataFrame, info: pl.DataFrame) -> pl.DataFrame:
    """
    Compute sector dummies.
    """
    df = df.join(info.select(["tic", "sector"]), on="tic", how="left")
    df = df.filter(pl.col("sector").is_in(config.processing.sectors))
    df = df.to_dummies(columns=["sector"])
    df = df.sort(["tic", "rdq"])
    df = df.with_columns([pl.col(c).cast(pl.Int8) for c in df.columns if c.startswith("sector_")])
    df = df.rename(
        {col: col.lower().replace(" ", "_") for col in df.columns if col.startswith("sector_")}
    )
    return df
