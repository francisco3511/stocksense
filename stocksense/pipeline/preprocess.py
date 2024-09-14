import polars as pl
import numpy as np
import datetime as dt

from config import get_config
from database_handler import DatabaseHandler

import talib as ta

YEARLY_TRADING_DAYS = 252
QUARTERLY_TRADING_DAYS = 60


class Preprocess():
    """
    Data preprocessing handling class.
    """

    def __init__(self):
        self.db = DatabaseHandler()
        self.features = get_config("data")["features"]
        self.targets = get_config("data")["targets"]
        self.index_data = None
        self.data = None

    def process_data(self):
        """
        Computes financial features for all stocks with
        financial observations recorded on the database.
        """
        # compute index forward returns
        self.index_data = self.process_index_data()
        # compute all features
        self.data = self.feature_engineering()

    def process_index_data(self) -> pl.DataFrame:
        # load index data
        index_df = self.db.fetch_index_data()

        # parse date
        index_df = index_df.with_columns(
            pl.col('date').str.strptime(pl.Date, format='%Y-%m-%d')
        )
        # compute index forward returns
        index_df = index_df.with_columns(
            ((pl.col('adj_close').shift(-YEARLY_TRADING_DAYS) / pl.col('adj_close')) - 1).alias('index_freturn')
        )
        return index_df

    def feature_engineering(self) -> pl.DataFrame:
        """
        Compute financial ratios and features for training.
        """
        # get financial data
        df = self.db.fetch_financial_data()

        df = df.with_columns(
            pl.col('rdq').shift().over('tic').alias('prev_rdq')
        ).sort(['tic', 'rdq'])

        df = df.filter(pl.col('tic').is_in(["AMZN", "AAPL"]))

        # fetch ALL other stock data from source tables
        info = self.db.fetch_info()
        market_df = self.db.fetch_market_data().sort(['tic', 'date'])
        insider_df = self.db.fetch_insider_data()

        # detect splits and adjust data
        df = adjust_shares(df)

        # compute insider trading counts
        df = compute_insider_trading_features(df, insider_df)

        # compute financial ratios
        df = df.groupby('tic').apply(lambda x: compute_financial_ratios(x)).reset_index(drop=True)

        # compute growth metrics
        df = df.groupby('tic').apply(lambda x: compute_growth_ratios(x)).reset_index(drop=True)

        # compute price/market ratios
        df = df.groupby('tic').apply(lambda x: compute_market_ratios(x, x['tic'][0], market_df, self.index_data)).reset_index(drop=True)

        # add id info and select relevant features
        df = df.join(info.select(['tic', 'sector']), on='tic', how='left')

        df = df.select(
            ['obs_date', 'tic', 'sector', 'close'] + self.features + self.targets
        )
        return df

    def save_data(self, name: str) -> None:
        self.data.write_csv(f"data/2_production_data/{name}.csv")


def map_to_closest_split_factor(approx_factor):
    common_split_ratios = np.array([
        1, 0.5, 0.33, 0.25,
        2, 3, 4, 5, 6, 7,
        8, 9, 10, 15, 20, 30
    ])
    # compute the absolute differences
    diffs = np.abs(common_split_ratios - approx_factor)
    # find the index of the minimum difference
    closest_index = np.argmin(diffs)
    # return the closest common split factor
    return common_split_ratios[closest_index]


def adjust_shares(df):
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
    df = df.with_columns(
        pl.col('cshoq').fill_null(strategy='backward').over('tic')
    ).with_columns(
        (pl.col('cshoq') / pl.col('cshoq').shift()).over('tic').alias('csho_ratio')
    ).with_columns(
        ((pl.col('cshoq') / pl.col('cshoq').shift() - 1) > 0.25).alias('stock_split')
    ).with_columns(
        pl.col('csho_ratio').map_elements(
            map_to_closest_split_factor,
            return_dtype=pl.Float64
        ).alias('split_factor')
    )

    # compute the adjustment factor: 1 if no split occurred, otherwise the split factor
    df = df.with_columns(
        pl.when(pl.col("stock_split"))
        .then(pl.col("split_factor"))
        .otherwise(1.0)
        .shift(-1)
        .fill_null(strategy='backward')
        .alias("adjustment_factor")
    )
    df = df

    # compute cumulative product of adjustment factors in reverse (from latest to earliest)
    df = df.sort(by=["tic", "datadate"]).with_columns([
        pl.col("adjustment_factor")
        .cum_prod(reverse=True)
        .over("tic")
        .alias("cum_adjustment_factor")
    ])

    # apply the cumulative adjustment to the financial data
    df = df.with_columns(
        (pl.col("cshoq") * pl.col("cum_adjustment_factor"))
        .alias("cshoq")
    )

    # sort back to original order
    df = df.sort(by=["tic", "datadate"])
    return df.drop(['csho_ratio', 'split_factor', 'adjustment_factor', 'cum_adjustment_factor'])


def compute_insider_trading_features(
        df: pl.DataFrame,
        insider_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Computes insider trading features
    (i.e., number of insider purchases throughout the past quarter).

    Parameters
    ----------
    df : pl.DataFrame
        Financial data of a given stock.
    insider_df : pl.DataFrame
        Global insider trading data.

    Returns
    -------
    pl.DataFrame
        Data with insider trading features.
    """
    try:
        # filter purchases
        insider_purchases = insider_df.filter(
            pl.col("transaction_type") == "P - Purchase"
        )
        # cross join and filter
        df_cross = insider_purchases.join(df, how="cross")
        df_filtered = df_cross.filter(
            (pl.col("filling_date") <= pl.col("rdq")) &
            (pl.col("filling_date") >= pl.col("rdq") - dt.timedelta(days=90)) &
            (pl.col("tic") == pl.col("tic_right"))
        )
        # count the number of events for each observation date
        df_event_counts = df_filtered.group_by(["rdq", "tic"]).agg([
            pl.col("filling_date").count().alias("n_purchases")
        ])
        # join the count result back to the original observations dataframe
        df = df.join(df_event_counts, on=["rdq", "tic"], how="left").fill_null(0)

    except Exception:
        df = df.with_columns(pl.lit(np.nan).alias('n_purchases'))

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
    return df.with_columns([
        (pl.col('niq').rolling_sum(window_size=4) / pl.col('atq').rolling_mean(window_size=2)).alias('roa'),
        (pl.col('niq').rolling_sum(window_size=4) / pl.col('seqq')).alias('roe'),
        ((pl.col('saleq') - pl.col('cogsq')) / pl.col('saleq')).alias('gpm'),
        (pl.col('ebitdaq') / pl.col('saleq')).alias('ebitdam'),
        (pl.col('oancfq') / pl.col('saleq')).alias('cfm'),
        (pl.col('actq') / pl.col('lctq')).alias('cr'),
        ((pl.col('rectq') + pl.col('cheq')) / pl.col('lctq')).alias('qr'),
        (pl.col('cheq') / pl.col('lctq')).alias('csr'),
        (pl.col('ltq') / pl.col('atq')).alias('dr'),
        (pl.col('ltq') / pl.col('seqq')).alias('der'),
        (pl.col('ltq') / pl.col('ebitdaq')).alias('debitda'),
        (pl.col('dlttq') / pl.col('atq')).alias('ltda'),
        ((pl.col('oancfq') - pl.col('capxq')) / pl.col('dlttq')).alias('ltcr'),
        (pl.col('saleq') / pl.col('invtq').rolling_mean(window_size=2)).alias('itr'),
        (pl.col('saleq') / pl.col('rectq').rolling_mean(window_size=2)).alias('rtr'),
        (pl.col('saleq') / pl.col('atq').rolling_mean(window_size=2)).alias('atr')
    ])


def compute_growth_ratios(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute rolling growth statistics.

    Parameters
    ----------
    df : pl.DataFrame
        Financial data of a given stock.

    Returns
    -------
    pl.DataFrame
        Data with additional columns.
    """
    df = df.with_columns([
        ((pl.col('niq') - pl.col('niq').shift(1)) / pl.col('niq').shift(1)).alias('ni_qoq'),
        ((pl.col('niq') - pl.col('niq').shift(4)) / pl.col('niq').shift(4)).alias('ni_yoy'),
        ((pl.col('niq') - pl.col('niq').shift(8)) / pl.col('niq').shift(8)).alias('ni_2y'),
        ((pl.col('saleq') - pl.col('saleq').shift(4)) / pl.col('saleq').shift(4)).alias('rev_yoy'),
        ((pl.col('dlttq') - pl.col('dlttq').shift(4)) / pl.col('dlttq').shift(4).abs()).alias('ltd_yoy'),
        ((pl.col('dr') - pl.col('dr').shift(4)) / pl.col('dr').shift(4).abs()).alias('dr_yoy')
    ])
    return df

def compute_market_ratios(df: pl.DataFrame, tic: str, market_df: pl.DataFrame, index_df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute market-related ratios.

    Parameters
    ----------
    df : pl.DataFrame
        Financial data of a given stock.
    tic : str
        Stock ticker.
    market_df : pl.DataFrame
        Market data.
    index_df : pl.DataFrame
        Index price data.

    Returns
    -------
    pl.DataFrame
        Main dataset with added ratios.
    """
    market_df = market_df.filter(pl.col('tic') == tic)

    market_df = market_df.with_columns([
        ((pl.col('adj_close').shift(-YEARLY_TRADING_DAYS) / pl.col('adj_close')) - 1).alias('freturn'),
        (ta.RSI(market_df['close'], timeperiod=9)).alias('rsi_9d'),
        (ta.RSI(market_df['close'], timeperiod=30)).alias('rsi_30d'),
        ((pl.col('close') - pl.col('close').shift(YEARLY_TRADING_DAYS)) / pl.col('close').shift(YEARLY_TRADING_DAYS)).alias('price_yoy'),
        ((pl.col('close') - pl.col('close').shift(QUARTERLY_TRADING_DAYS)) / pl.col('close').shift(QUARTERLY_TRADING_DAYS)).alias('price_qoq')
    ])

    df = df.join_asof(market_df.drop(['tic', 'volume']), on='rdq', by='date', strategy='forward', tolerance=dt.timedelta(days=7))
    df = df.join_asof(index_df.select(['date', 'index_freturn']), on='rdq', by='date', strategy='forward', tolerance=dt.timedelta(days=7))

    df = df.with_columns([
        pl.col('date').alias('obs_date'),
        (pl.col('freturn') - pl.col('index_freturn')).alias('adj_freturn'),
        (pl.col('adj_freturn') > 0).cast(pl.Int8).alias('adj_fperf'),
        (pl.col('adj_freturn') > 0.2).cast(pl.Int8).alias('adj_fperf_20'),
        (pl.col('niq').rolling_sum(window_size=4) / pl.col('adj_shares')).alias('eps'),
        (pl.col('close') / pl.col('eps')).alias('pe'),
        (pl.col('close') * pl.col('adj_shares')).alias('mkt_cap'),
        (pl.col('mkt_cap') + pl.col('ltq') - pl.col('cheq')).alias('ev'),
        (pl.col('ev') / pl.col('ebitdaq').rolling_sum(window_size=4)).alias('ev_ebitda')
    ])
    return df
