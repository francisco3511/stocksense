import polars as pl
import numpy as np
import datetime as dt
import polars_talib as plta
from loguru import logger
from pathlib import Path

from config import get_config
from database_handler import DatabaseHandler

CONFIG = get_config("processing")
PACKAGE_DIR = Path(__file__).parents[1]
DATA_PATH = PACKAGE_DIR / "data"


class Preprocess:
    """
    Stock data processing pipeline handler.
    """

    def __init__(self):
        self.db = DatabaseHandler()
        self.index_data = self._process_index_data()

    def run(self):
        """
        Runs main data processing pipeline.
        """
        logger.info("START processing stock data")
        data = self.feature_engineering()
        data = self.clean_data(data)
        logger.success("END processing stock data")
        return data

    def _process_index_data(self) -> pl.DataFrame:
        """
        Process S&P500 index price data.

        Returns
        -------
        pl.DataFrame
            Processed data, incl. forward and past return rates.
        """
        logger.info("START processing S&P500 index data")

        index_df = self.db.fetch_index_data()
        if index_df.is_empty():
            raise ValueError("Empty index data received")

        index_df = index_df.sort(by=["date"])

        # compute index past returns
        index_df = index_df.with_columns(
            [
                pl.col("close")
                .pct_change(CONFIG["month_trading_days"])
                .alias("index_mom"),
                pl.col("close")
                .pct_change(CONFIG["quarter_trading_days"])
                .alias("index_qoq"),
                pl.col("close")
                .pct_change(CONFIG["semester_trading_days"])
                .alias("index_sos"),
                pl.col("close")
                .pct_change(CONFIG["year_trading_days"])
                .alias("index_yoy"),
                pl.col("close")
                .pct_change(CONFIG["2year_trading_days"])
                .alias("index_2y"),
            ]
        )

        # compute volatily of index
        index_df = index_df.with_columns(
            (pl.col("close") / pl.col("close").shift(1)).log().alias("log_return")
        ).with_columns(
            pl.col("log_return")
            .rolling_std(CONFIG["month_trading_days"])
            .alias("index_vol_mom"),
            pl.col("log_return")
            .rolling_std(CONFIG["quarter_trading_days"])
            .alias("index_vol_qoq"),
            pl.col("log_return")
            .rolling_std(CONFIG["semester_trading_days"])
            .alias("index_vol_sos"),
            pl.col("log_return")
            .rolling_std(CONFIG["year_trading_days"])
            .alias("index_vol_yoy"),
            pl.col("log_return")
            .rolling_std(CONFIG["2year_trading_days"])
            .alias("index_vol_2y"),
        )

        index_df = index_df.rename(
            {
                "date": "index_date",
                "close": "index_close",
                "adj_close": "index_adj_close",
            }
        )

        logger.success(f"S&P500 index data {index_df.shape[0]} rows PROCESSED")

        return index_df.select(
            [
                "index_date",
                "index_close",
                "index_adj_close",
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

    def feature_engineering(self) -> pl.DataFrame:
        """
        Compute financial ratios and features for training.
        """
        logger.info("START feature engineering")

        # fetch data
        df = self.db.fetch_financial_data()
        info = self.db.fetch_stock()
        market_df = self.db.fetch_market_data().sort(["tic", "date"])
        insider_df = self.db.fetch_insider_data()

        # compute all features
        df = compute_trade_date(df)
        df = adjust_shares(df)
        df = compute_insider_trading_features(df, insider_df)
        df = compute_financial_ratios(df)
        df = compute_market_ratios(df, market_df, self.index_data)
        df = compute_growth_ratios(df)
        df = compute_performance_targets(df)
        df = df.join(info.select(["tic", "sector"]), on="tic", how="left")

        logger.success(f"{df.shape[1]} features PROCESSED")
        return df

    def clean_data(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Clean and process financial features dataset.

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

        df = data.filter(pl.col("tdq") <= pl.lit(dt.datetime.today().date()))
        growth_alias = ["qoq", "yoy", "2y", "return"]
        growth_vars = [f for f in df.columns if any(xf in f for xf in growth_alias)]
        df = df.filter(~pl.all_horizontal(pl.col("niq_2y").is_null()))
        df = df.filter(
            pl.col("sector").is_in(
                [
                    "Health Care",
                    "Financials",
                    "Industrials",
                    "Consumer Discretionary",
                    "Information Technology",
                    "Communication Services",
                    "Consumer Staples",
                    "Utilities",
                    "Real Estate",
                    "Materials",
                    "Energy",
                ]
            )
        )

        for feature in [f for f in df.columns if any(xf in f for xf in growth_vars)]:
            df = df.with_columns(df.with_columns(pl.col(feature).clip(-30, 30)))

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

        df = df.with_columns([pl.col("freturn") * 100, pl.col("adj_freturn") * 100])
        df = df.to_dummies(columns=["sector"])

        logger.success(f"{df.shape[0]} rows retained after CLEANING")
        return df

    def save_data(self, data: pl.DataFrame) -> None:
        """
        Saves processed data locally.

        Parameters
        ----------
        data : pl.DataFrame
            Filtered and processed data.
        """
        today = dt.datetime.today().date()
        file_name = f"data/processed/proc_{today}.csv"
        data.write_csv(file_name)


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


def compute_trade_date(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute trade dates.
    """

    min_year = df["rdq"].dt.year().min()
    max_year = df["rdq"].dt.year().max()

    quarter_dates = generate_quarter_dates(min_year, max_year)
    quarter_df = pl.DataFrame({"tdq": quarter_dates}).with_columns(
        pl.col("tdq").dt.date()
    )

    df = df.sort(by=["rdq", "tic"])
    df = df.join_asof(quarter_df, left_on="rdq", right_on="tdq", strategy="forward")
    return df.sort(by=["tic", "rdq"])


def map_to_closest_split_factor(approx_factor: float) -> float:
    common_split_ratios = np.array(
        [1, 0.5, 0.33, 0.25, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]
    )
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
        ((pl.col("cshoq") / pl.col("cshoq").shift() - 1) > 0.25)
        .over("tic")
        .alias("stock_split")
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

    # compute cumulative product of adjustment factors in reverse (from latest to earliest)
    df = df.sort(by=["tic", "datadate"]).with_columns(
        pl.col("adjustment_factor")
        .cum_prod(reverse=True)
        .over("tic")
        .alias("cum_adjustment_factor")
    )

    # apply the cumulative adjustment to the financial data
    df = df.with_columns(
        (pl.col("cshoq") * pl.col("cum_adjustment_factor")).alias("cshoq")
    )

    df = df.sort(by=["tic", "tdq"])
    return df.drop(
        ["csho_ratio", "split_factor", "adjustment_factor", "cum_adjustment_factor"]
    )


def compute_insider_purchases(
    df: pl.DataFrame, insider_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Computes insider purchases quarterly features.
    """

    insider_filtered_lazy = insider_df.lazy().filter(
        (pl.col("transaction_type") == "P - Purchase")
    )
    df_lazy = df.lazy()

    df_filtered_lazy = df_lazy.join(
        insider_filtered_lazy, how="inner", left_on=["tic"], right_on=["tic"]
    ).filter(
        (pl.col("filling_date") < pl.col("tdq"))
        & (pl.col("filling_date") >= pl.col("tdq").dt.offset_by("-12mo"))
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
        (pl.col("transaction_type") == "S - Sale")
        | (pl.col("transaction_type") == "S - Sale+OE")
    )
    df_lazy = df.lazy()

    df_filtered_lazy = df_lazy.join(
        insider_filtered_lazy, how="inner", left_on=["tic"], right_on=["tic"]
    ).filter(
        (pl.col("filling_date") < pl.col("tdq"))
        & (pl.col("filling_date") >= pl.col("tdq").dt.offset_by("-12mo"))
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


def compute_insider_trading_features(
    df: pl.DataFrame, insider_df: pl.DataFrame
) -> pl.DataFrame:
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
    df = df.with_columns(
        (pl.col("val_sales") - pl.col("val_purch")).alias("insider_balance")
    )
    return df


def compute_financial_ratios(df: pl.DataFrame) -> pl.DataFrame:
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
    return (
        df.lazy()
        .with_columns(
            (pl.col("niq").rolling_sum(4) / pl.col("atq").rolling_mean(2))
            .over("tic")
            .alias("roa"),
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
            (pl.col("saleq") / pl.col("invtq").rolling_mean(2))
            .over("tic")
            .alias("itr"),
            (pl.col("saleq") / pl.col("rectq").rolling_mean(2))
            .over("tic")
            .alias("rtr"),
            (pl.col("saleq") / pl.col("atq").rolling_mean(2)).over("tic").alias("atr"),
            pl.col("atq").log().alias("size"),
        )
        .collect()
    )


def compute_market_ratios(
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

    market_df = compute_daily_momentum_features(market_df)
    market_df = compute_daily_volatility_features(market_df)

    df = df.sort(by=["rdq", "tic"])
    df = (
        df.join_asof(
            market_df.drop(["volume"]),
            left_on="tdq",
            right_on="date",
            by="tic",
            strategy="backward",
            tolerance=dt.timedelta(days=7),
        )
        .join_asof(
            market_df.select(["date", "tic", "close"]).rename({"close": "rdq_close"}),
            left_on="rdq",
            right_on="date",
            by="tic",
            strategy="forward",
            tolerance=dt.timedelta(days=7),
        )
        .join_asof(
            index_df,
            left_on="tdq",
            right_on="index_date",
            strategy="backward",
            tolerance=dt.timedelta(days=7),
        )
    )
    df = df.sort(by=["tic", "tdq"])

    df = compute_hybrid_features(df)
    return df


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
        [
            plta.rsi(pl.col("close"), timeperiod=14).over("tic").alias("rsi_14d"),
            plta.rsi(pl.col("close"), timeperiod=30).over("tic").alias("rsi_30d"),
            plta.rsi(pl.col("close"), timeperiod=60).over("tic").alias("rsi_60d"),
            plta.rsi(pl.col("close"), timeperiod=90).over("tic").alias("rsi_90d"),
            plta.rsi(pl.col("close"), timeperiod=360).over("tic").alias("rsi_1y"),
            pl.col("close")
            .pct_change(CONFIG["month_trading_days"])
            .over("tic")
            .alias("price_mom"),
            pl.col("close")
            .pct_change(CONFIG["quarter_trading_days"])
            .over("tic")
            .alias("price_qoq"),
            pl.col("close")
            .pct_change(CONFIG["year_trading_days"])
            .over("tic")
            .alias("price_yoy"),
            pl.col("close")
            .pct_change(CONFIG["2year_trading_days"])
            .over("tic")
            .alias("price_2y"),
        ]
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
        (pl.col("close") / pl.col("close").shift(1))
        .log()
        .over("tic")
        .alias("log_return")
    )
    return df.with_columns(
        pl.col("log_return")
        .rolling_std(CONFIG["month_trading_days"])
        .over("tic")
        .alias("vol_mom"),
        pl.col("log_return")
        .rolling_std(CONFIG["quarter_trading_days"])
        .over("tic")
        .alias("vol_qoq"),
        pl.col("log_return")
        .rolling_std(CONFIG["year_trading_days"])
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
            (pl.col("mkt_cap") / pl.col("saleq").rolling_sum(4))
            .over("tic")
            .alias("ps"),
        )
        .with_columns(
            (pl.col("ev") / pl.col("ebitdaq").rolling_sum(4))
            .over("tic")
            .alias("ev_ebitda")
        )
    )


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

    df = df.with_columns(
        (
            (
                pl.col("index_adj_close").shift(-CONFIG["prediction_horizon"])
                / pl.col("index_adj_close")
            )
            - 1
        )
        .over("tic")
        .alias("index_freturn"),
        (
            (
                pl.col("adj_close").shift(-CONFIG["prediction_horizon"])
                / pl.col("adj_close")
            )
            - 1
        )
        .over("tic")
        .alias("freturn"),
        ((pl.col("adj_close").shift(-1) / pl.col("adj_close")) - 1)
        .over("tic")
        .alias("freturn_1q"),
        ((pl.col("adj_close").shift(-2) / pl.col("adj_close")) - 1)
        .over("tic")
        .alias("freturn_2q"),
        ((pl.col("adj_close").shift(-3) / pl.col("adj_close")) - 1)
        .over("tic")
        .alias("freturn_3q"),
        ((pl.col("adj_close").shift(-4) / pl.col("adj_close")) - 1)
        .over("tic")
        .alias("freturn_4q"),
    )
    df = df.with_columns(
        (pl.col("freturn") - pl.col("index_freturn")).alias("adj_freturn")
    )
    df = df.with_columns(
        (pl.col("adj_freturn") > CONFIG["over_performance_threshold"])
        .cast(pl.Int8)
        .alias("adj_fperf"),
        (pl.col("freturn_1q") > CONFIG["performance_threshold"])
        .cast(pl.Int8)
        .alias("fperf_1q"),
        (pl.col("freturn_2q") > CONFIG["performance_threshold"])
        .cast(pl.Int8)
        .alias("fperf_2q"),
        (pl.col("freturn_3q") > CONFIG["performance_threshold"])
        .cast(pl.Int8)
        .alias("fperf_3q"),
        (pl.col("freturn_4q") > CONFIG["performance_threshold"])
        .cast(pl.Int8)
        .alias("fperf_4q"),
    ).with_columns(
        (
            (
                pl.col("fperf_1q")
                + pl.col("fperf_2q")
                + pl.col("fperf_3q")
                + pl.col("fperf_4q")
            )
            > 0
        )
        .cast(pl.Int8)
        .alias("fperf")
    )
    return df


def _compute_growth_rate(col: str, periods: int, suffix: str) -> pl.Expr:
    """
    Compute growth rate for a given column over specified periods.

    Parameters
    ----------
    col : str
        Column name to compute growth for
    periods : int
        Number of periods to shift (e.g., 1 for QoQ, 4 for YoY)
    suffix : str
        Suffix for the output column name (e.g., 'qoq', 'yoy')

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


def _compute_signed_growth_rate(col: str, periods: int, suffix: str) -> pl.Expr:
    """
    Compute signed growth rate for columns that can be negative.

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
        Polars expression for the signed growth calculation
    """
    return (
        (
            (pl.col(col) - pl.col(col).shift(periods)).sign()
            * (pl.col(col) - pl.col(col).shift(periods))
            / pl.col(col).shift(periods).abs()
        )
        .over("tic")
        .alias(f"{col}_{suffix}")
    )


def compute_growth_ratios(df: pl.DataFrame) -> pl.DataFrame:
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

    QUARTER = 1
    YEAR = 4
    TWO_YEAR = 8

    signed_metrics = {"niq": [YEAR, TWO_YEAR]}  # Can have negative values

    standard_metrics = {
        "saleq": [YEAR, TWO_YEAR],
        "ltq": [QUARTER, YEAR, TWO_YEAR],
        "dlttq": [YEAR],
        "gpm": [QUARTER, YEAR],
        "roa": [QUARTER, YEAR],
        "roe": [QUARTER, YEAR],
        "fcf": [QUARTER, YEAR],
        "cr": [QUARTER, YEAR],
        "qr": [QUARTER, YEAR],
        "der": [QUARTER, YEAR],
        "dr": [QUARTER, YEAR],
        "ltda": [YEAR],
        "pe": [QUARTER, YEAR],
        "pb": [QUARTER, YEAR],
        "ps": [QUARTER, YEAR],
        "eps": [QUARTER, YEAR],
        "ev_ebitda": [QUARTER, YEAR],
        "ltcr": [YEAR],
        "itr": [YEAR],
        "rtr": [YEAR],
        "atr": [YEAR],
    }

    expressions = []

    # add signed growth calculations
    for metric, periods in signed_metrics.items():
        for period in periods:
            suffix = "qoq" if period == QUARTER else "yoy" if period == YEAR else "2y"
            expressions.append(_compute_signed_growth_rate(metric, period, suffix))

    # add standard growth calculations
    for metric, periods in standard_metrics.items():
        for period in periods:
            suffix = "qoq" if period == QUARTER else "yoy" if period == YEAR else "2y"
            expressions.append(_compute_growth_rate(metric, period, suffix))

    return df.lazy().with_columns(expressions).collect()
