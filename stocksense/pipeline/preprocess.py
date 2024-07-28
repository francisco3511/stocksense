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
        self.db_handler = DatabaseHandler()
        self.features = get_config("data")["features"]
        self.targets = get_config("data")["targets"]
        self.index_data = None
        self.data = None

    def process_data(self):
        """
        Computes financial features for all stocks with
        financial observations recorded on database.
        """
        # compute index forward returns
        self.index_data = self.process_index_data()
        # compute all features
        self.data = self.feature_engineering()

    def process_index_data(self) -> pl.DataFrame:
        """
        Processes index price data.

        Returns
        -------
        pl.DataFrame
            Daily index price data with computed features.
        """
        # load index data
        index_data = self.db_handler.fetch_sp_data()

        # parse date
        index_data = index_data.with_columns(
            pl.col('date').str.strptime(pl.Date, format='%Y-%m-%d')
        )

        # compute index forward returns
        index_data = index_data.with_columns(
            (pl.col('adj_close').shift(-YEARLY_TRADING_DAYS) / pl.col('adj_close') - 1).alias('index_freturn')
        )

        return index_data

    def feature_engineering(self) -> pl.DataFrame:
        """
        Compute financial ratios and features for training.
        """
        # get financial data
        df = self.db_handler.fetch_financial_data()

        df = df.with_columns(
            pl.col('rdq').str.strptime(pl.Date, format='%Y-%m-%d')
        ).with_columns(
            pl.col('rdq').shift(1).alias('prev_rdq')
        ).sort(['tic', 'rdq'])

        # fetch ALL other stock data from source tables
        info = self.db_handler.fetch_stock_info()
        market_df = self.db_handler.fetch_market_data()
        insider_df = self.db_handler.fetch_insider_data()

        # convert and sort market data
        market_df = market_df.with_columns(
            pl.col('date').str.strptime(pl.Date, format='%Y-%m-%d')
        ).sort(['tic', 'date'])

        # compute curr shares outstanding
        df = df.groupby('tic').apply(lambda x: compute_shares(x)).reset_index(drop=True)

        # compute insider trading counts
        df = df.groupby('tic').apply(lambda x: compute_insider_trading_features(x, x['tic'][0], insider_df)).reset_index(drop=True)

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

def compute_shares(df: pl.DataFrame) -> pl.DataFrame:
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
        Data with adjusted outstd. shares.
    """
    df = df.with_columns(
        pl.col('cshoq').fill_none(strategy='backward')
    ).with_columns(
        (pl.col('cshoq') / pl.col('cshoq').shift(1) - 1).alias('pct_change')
    ).with_columns(
        (pl.col('pct_change').abs() > 0.25).alias('stock_split')
    ).with_columns(
        pl.col('cshoq').alias('adj_shares')
    )

    common_split_ratios = [
        0.5, 0.33, 0.25, 
        2, 3, 4, 5, 6, 7, 8, 10, 20, 30
    ]

    for i in range(len(df)):
        if df[i, 'stock_split']:
            split_ratio = df[i, 'cshoq'] / df[i - 1, 'cshoq']
            closest_split = min(common_split_ratios, key=lambda x: abs(x - split_ratio))
            if abs(closest_split - split_ratio) < 0.1:
                split_ratio = float(closest_split)
            df[:i, 'adj_shares'] *= split_ratio

    return df.drop(['stock_split', 'pct_change'])

def compute_insider_trading_features(df: pl.DataFrame, tic: str, insider_df: pl.DataFrame) -> pl.DataFrame:
    """
    Computes insider trading features
    (i.e. number of insider purchases throughout past quarter)

    Parameters
    ----------
    df : pl.DataFrame
        Financial data of a given stock.
    tic : str
        Stock ticker.
    insider_df : pl.DataFrame
        Global insider trading data.

    Returns
    -------
    pl.DataFrame
        Data with insider trading features.
    """
    try:
        insider_df = insider_df.filter(pl.col('tic') == tic)

        df = df.with_columns(pl.lit(1).alias('key'))
        insider_df = insider_df.with_columns(pl.lit(1).alias('key'))
        merged_df = df.join(insider_df, on='key').drop('key')

        count_p = merged_df.filter(
            (pl.col('filling_date') >= pl.col('prev_rdq')) &
            (pl.col('filling_date') < pl.col('rdq')) &
            (pl.col('transaction_type') == 'P - Purchase')
        ).groupby('rdq').agg(pl.count('filling_date').alias('n_purchases'))

        df = df.join(count_p, on='rdq', how='left')
        df = df.with_columns(pl.col('n_purchases').fill_none(0))
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
    df = df.with_columns([
        (pl.col('niq').rolling_sum(4) / pl.col('atq').rolling_mean(2)).alias('roa'),
        (pl.col('niq').rolling_sum(4) / pl.col('seqq')).alias('roe'),
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
        (pl.col('saleq') / pl.col('invtq').rolling_mean(2)).alias('itr'),
        (pl.col('saleq') / pl.col('rectq').rolling_mean(2)).alias('rtr'),
        (pl.col('saleq') / pl.col('atq').rolling_mean(2)).alias('atr')
    ])
    return df

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
    Compute market related ratios.

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
        (pl.col('adj_close').shift(-YEARLY_TRADING_DAYS) / pl.col('adj_close') - 1).alias('freturn'),
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
        (pl.col('niq').rolling_sum(4) / pl.col('adj_shares')).alias('eps'),
        (pl.col('close') / pl.col('eps')).alias('pe'),
        (pl.col('close') * pl.col('adj_shares')).alias('mkt_cap'),
        (pl.col('mkt_cap') + pl.col('ltq') - pl.col('cheq')).alias('ev'),
        (pl.col('ev') / pl.col('ebitdaq').rolling_sum(4)).alias('ev_ebitda')
    ])
    return df
