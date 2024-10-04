import polars as pl
import numpy as np
import datetime as dt

from config import get_config
from database_handler import DatabaseHandler

import polars_talib as plta

YEARLY_TRADING_DAYS = 252
QUARTERLY_TRADING_DAYS = 60
MONTHLY_TRADING_DAYS = 21

# TODO: add logging and exception handling


class Preprocess():
    """
    Data preprocessing handling class.
    """

    def __init__(self):
        self.db = DatabaseHandler()
        self.features = get_config('model')['features']
        self.targets = get_config('model')['targets']
        self.index_data = None
        self.data = None

    def process_data(self):
        """
        Computes financial features for all stocks with
        financial observations recorded on the database.
        """
        # compute index features
        self.index_data = self.process_index_data()
        # compute all features
        self.data = self.feature_engineering()

    def process_index_data(self) -> pl.DataFrame:
        """
        Process S&P500 index price data.

        Returns
        -------
        pl.DataFrame
            Processed data, incl. forward and past return rates.
        """
        # load index data
        index_df = self.db.fetch_index_data()

        # parse date
        index_df = index_df.with_columns(
            pl.col('date').str.strptime(pl.Date, format="%Y-%m-%d")
        )
        # compute index forward returns
        index_df = index_df.with_columns(
            pl.col('adj_close').pct_change(-YEARLY_TRADING_DAYS).alias('index_freturn'),
            pl.col('adj_close').pct_change(QUARTERLY_TRADING_DAYS).alias('index_qoq'),
            pl.col('adj_close').pct_change(YEARLY_TRADING_DAYS).alias('index_yoy')
        )
        return index_df

    def feature_engineering(self) -> pl.DataFrame:
        """
        Compute financial ratios and features for training.
        """
        # get financial data
        df = self.db.fetch_financial_data()

        df = df.filter(pl.col('tic').is_in(['AAPL']))

        # fetch ALL other stock data from source tables
        info = self.db.fetch_stock()
        market_df = self.db.fetch_market_data().sort(['tic', 'date'])
        insider_df = self.db.fetch_insider_data()

        # compute trade dates
        df = compute_trade_date(df)

        # detect splits and adjust data
        df = adjust_shares(df)

        # compute insider trading features
        df = compute_insider_purchases(df, insider_df)
        df = compute_insider_sales(df, insider_df)

        # compute financial ratios
        df = compute_financial_ratios(df)

        # compute price/market ratios
        df = compute_market_ratios(df, market_df, self.index_data)

        # compute growth metrics
        df = compute_growth_ratios(df)

        # add id info and select relevant features
        df = df.join(info.select(['tic', 'sector']), on='tic', how='left')

        df = df.select(
            ['datadate', 'rdq', 'tdq', 'tic', 'sector'] +
            self.features +
            self.targets
        )
        return df

    def save_data(self, name: str) -> None:
        self.data.write_csv(f"data/2_production_data/{name}.csv")


def generate_quarter_dates(start_year: int, end_year: int) -> pl.DataFrame:
    """
    Function to generate quarter-end dates for a range of years.
    """
    quarter_end_dates = []
    for year in range(start_year, end_year + 1):
        quarter_end_dates.extend([
            dt.datetime(year, 3, 1),
            dt.datetime(year, 6, 1),
            dt.datetime(year, 9, 1),
            dt.datetime(year, 12, 1),
        ])
    return quarter_end_dates


def compute_trade_date(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute trade dates.
    """

    # get the range of years in the original dataframe
    min_year = df['rdq'].min().year
    max_year = df['rdq'].max().year

    # generate all quarter-end dates for the relevant years
    quarter_dates = generate_quarter_dates(min_year, max_year)

    # create a Polars DataFrame for the quarter-end dates
    quarter_df = pl.DataFrame({
        'tdq': quarter_dates
    }).with_columns(pl.col('tdq').dt.date())

    # join where each date is matched with the next available quarter-end
    df = df.join_asof(
        quarter_df,
        left_on='rdq',
        right_on='tdq',
        strategy='forward'
    )

    return df


def map_to_closest_split_factor(approx_factor: float) -> float:
    common_split_ratios = np.array([
        1, 0.5, 0.33, 0.25,
        2, 3, 4, 5, 6, 7,
        8, 9, 10, 15, 20, 30
    ])
    # compute the absolute differences and find id with min difference
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
        pl.when(pl.col('stock_split'))
        .then(pl.col('split_factor'))
        .otherwise(1.0)
        .shift(-1)
        .fill_null(strategy='forward')
        .alias('adjustment_factor')
    )

    # compute cumulative product of adjustment factors in reverse (from latest to earliest)
    df = df.sort(by=["tic", "datadate"]).with_columns([
        pl.col('adjustment_factor')
        .cum_prod(reverse=True)
        .over('tic')
        .alias('cum_adjustment_factor')
    ])

    # apply the cumulative adjustment to the financial data
    df = df.with_columns(
        (pl.col('cshoq') * pl.col('cum_adjustment_factor'))
        .alias('cshoq')
    )

    # sort back to original order
    df = df.sort(by=['tic', 'datadate'])
    return df.drop(['csho_ratio', 'split_factor', 'adjustment_factor', 'cum_adjustment_factor'])


def compute_insider_purchases(
    df: pl.DataFrame,
    insider_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Computes insider buying quarterly features,
    using SEC-4 fillings table.

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
    # Convert both dataframes to lazy mode
    df_lazy = df.lazy()
    insider_df_lazy = insider_df.lazy()

    # filter purchases
    insider_purchases_lazy = insider_df_lazy.filter(
        pl.col('transaction_type') == "P - Purchase"
    )

    # cross join and filter on the date ranges and ticker matches
    df_cross_lazy = insider_purchases_lazy.join(df_lazy, how='cross')
    df_filtered_lazy = df_cross_lazy.filter(
        (pl.col('filling_date') < pl.col('tdq')) &
        (pl.col('filling_date') >= pl.col('tdq').dt.offset_by('-3mo')) &
        (pl.col('tic') == pl.col('tic_right'))
    )

    # count the number and qty of purchases per trade period and firm
    df_event_counts = df_filtered_lazy.group_by(['tic', 'tdq']).agg([
        pl.col('filling_date').count().alias('n_purch')
    ])

    df_qty = df_filtered_lazy.group_by(['tic', 'tdq']).agg([
        (
            pl.col('value')
            .str.replace_all(r"[\$,€]", "")
            .str.replace_all(",", "")
            .cast(pl.Float64) / 1000000
        ).sum().round(3).alias('val_purch')
    ])

    # join back to the original dataframe, fill nulls and collect
    result = df_lazy.join(df_event_counts, on=['tic', 'tdq'], how='left').fill_null(0)
    result = result.join(df_qty, on=['tic', 'tdq'], how='left').fill_null(0)
    return result.collect()


def compute_insider_sales(
    df: pl.DataFrame,
    insider_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Computes insider selling quarterly features,
    using SEC-4 fillings table.

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
    # Convert both dataframes to lazy mode
    df_lazy = df.lazy()
    insider_df_lazy = insider_df.lazy()

    # filter purchases
    insider_sales_lazy = insider_df_lazy.filter(
        (pl.col('transaction_type') == "S - Sale") |
        (pl.col('transaction_type') == "S - Sale+OE")
    )

    # cross join and filter on the date ranges and ticker matches
    df_cross_lazy = insider_sales_lazy.join(df_lazy, how='cross')
    df_filtered_lazy = df_cross_lazy.filter(
        (pl.col('filling_date') < pl.col('tdq')) &
        (pl.col('filling_date') >= pl.col('tdq').dt.offset_by('-3mo')) &
        (pl.col('tic') == pl.col('tic_right'))
    )

    # count the number and qty of purchases per trade period and firm
    df_event_counts = df_filtered_lazy.group_by(['tic', 'tdq']).agg([
        pl.col('filling_date').count().alias('n_sales')
    ])

    df_qty = df_filtered_lazy.group_by(['tic', 'tdq']).agg([
        (
            - pl.col('value')
            .str.replace_all(r"[\$,€]", "")
            .str.replace_all(",", "")
            .cast(pl.Float64) / 1000000
        ).round(3).sum().alias('val_sales')
    ])

    # join back to the original dataframe, fill nulls and collect
    result = df_lazy.join(df_event_counts, on=['tdq', 'tic'], how='left').fill_null(0)
    result = result.join(df_qty, on=['tdq', 'tic'], how='left').fill_null(0)
    return result.collect()


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
    df = df.lazy().with_columns([
        (pl.col('niq') / pl.col('atq').rolling_mean(2).over('tic')).alias('roa'),
        (pl.col('niq') / pl.col('seqq')).alias('roe'),
        ((pl.col('saleq') - pl.col('cogsq')) / pl.col('saleq')).alias('gpm'),
        (pl.col('ebitdaq') / pl.col('saleq')).alias('ebitdam'),
        (pl.col('oancfq') / pl.col('saleq')).alias('cfm'),
        (pl.col('oancfq') - pl.col('capxq')).alias('fcf'),
        (pl.col('actq') / pl.col('lctq')).alias('cr'),
        ((pl.col('rectq') + pl.col('cheq')) / pl.col('lctq')).alias('qr'),
        (pl.col('cheq') / pl.col('lctq')).alias('csr'),
        (pl.col('ltq') / pl.col('atq')).alias('dr'),
        (pl.col('ltq') / pl.col('seqq')).alias('der'),
        (pl.col('ltq') / pl.col('ebitdaq')).alias('debitda'),
        (pl.col('dlttq') / pl.col('atq')).alias('ltda'),
        ((pl.col('oancfq') - pl.col('capxq')) / pl.col('dlttq')).alias('ltcr'),
        (pl.col('saleq') / pl.col('invtq').rolling_mean(2).over('tic')).alias('itr'),
        (pl.col('saleq') / pl.col('rectq').rolling_mean(2).over('tic')).alias('rtr'),
        (pl.col('saleq') / pl.col('atq').rolling_mean(2).over('tic')).alias('atr')
    ]).collect()
    return df


def compute_market_ratios(
    df: pl.DataFrame,
    market_df: pl.DataFrame,
    index_df: pl.DataFrame
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

    market_df = market_df.with_columns([
        pl.col('adj_close').pct_change(-YEARLY_TRADING_DAYS).over('tic').alias('freturn'),
        plta.rsi(pl.col('close'), timeperiod=14).over('tic').alias('rsi_14d'),
        plta.rsi(pl.col('close'), timeperiod=30).over('tic').alias('rsi_30d'),
        plta.rsi(pl.col('close'), timeperiod=60).over('tic').alias('rsi_60d'),
        pl.col('close').pct_change(YEARLY_TRADING_DAYS).over('tic').alias('price_yoy'),
        pl.col('close').pct_change(QUARTERLY_TRADING_DAYS).over('tic').alias('price_qoq'),
        (pl.col('close').log() - pl.col('close').shift(1).log()).alias('log_return')
    ])

    df = df.join_asof(
        market_df.drop(['volume']),
        left_on='tdq',
        right_on='date',
        by='tic',
        strategy='backward',
        tolerance=dt.timedelta(days=7)
    )
    df = df.join_asof(
        index_df.select(['date', 'adj_close', 'index_freturn', 'index_qoq', 'index_yoy']),
        left_on='tdq',
        right_on='date',
        by='date',
        strategy='backward',
        tolerance=dt.timedelta(days=7)
    )

    df = df.with_columns([
        pl.col('log_return').rolling_std(MONTHLY_TRADING_DAYS).alias('vol_m'),
        pl.col('log_return').rolling_std(QUARTERLY_TRADING_DAYS).alias('vol_q'),
        (pl.col('freturn') - pl.col('index_freturn')).alias('adj_freturn'),
        (pl.col('price_qoq') / pl.col('index_qoq')).alias('momentum_qoq'),
        (pl.col('price_yoy') / pl.col('index_yoy')).alias('momentum_yoy'),
        (pl.col('niq').rolling_sum(4).over('tic') / pl.col('cshoq')).alias('eps'),
    ]).with_columns([
        (pl.col('adj_freturn') > 0).cast(pl.Int8).alias('adj_fperf'),
        (pl.col('close') / pl.col('eps')).alias('pe'),
        (pl.col('close') * pl.col('cshoq')).alias('mkt_cap')
    ]).with_columns(
        (pl.col('mkt_cap') + pl.col('ltq') - pl.col('cheq')).alias('ev'),
    ).with_columns(
        (pl.col('ev') / pl.col('ebitdaq').rolling_sum(4).over('tic')).alias('ev_ebitda')
    )
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
    df = df.lazy().with_columns([
        pl.col('niq').pct_change(1).over('tic').alias('ni_qoq'),
        pl.col('niq').pct_change(4).over('tic').alias('ni_yoy'),
        pl.col('niq').pct_change(8).over('tic').alias('ni_2y'),
        pl.col('saleq').pct_change(1).over('tic').alias('rev_qoq'),
        pl.col('saleq').pct_change(4).over('tic').alias('rev_yoy'),
        pl.col('gpm').pct_change(1).over('tic').alias('gpm_qoq'),
        pl.col('gpm').pct_change(4).over('tic').alias('gpm_yoy'),
        pl.col('roa').pct_change(1).over('tic').alias('roa_qoq'),
        pl.col('roa').pct_change(4).over('tic').alias('roa_yoy'),
        pl.col('roe').pct_change(1).over('tic').alias('roe_qoq'),
        pl.col('roe').pct_change(4).over('tic').alias('roe_yoy'),
        pl.col('fcf').pct_change(1).over('tic').alias('fcf_qoq'),
        pl.col('fcf').pct_change(4).over('tic').alias('fcf_yoy'),
        pl.col('cr').pct_change(1).over('tic').alias('cr_qoq'),
        pl.col('cr').pct_change(4).over('tic').alias('cr_yoy'),
        pl.col('qr').pct_change(1).over('tic').alias('qr_qoq'),
        pl.col('qr').pct_change(4).over('tic').alias('qr_yoy'),
        pl.col('der').pct_change(1).over('tic').alias('der_qoq'),
        pl.col('der').pct_change(4).over('tic').alias('der_yoy'),
        pl.col('dr').pct_change(1).over('tic').alias('dr_qoq'),
        pl.col('dr').pct_change(4).over('tic').alias('dr_yoy'),
        pl.col('ltda').pct_change(4).over('tic').alias('ltda_yoy'),
        pl.col('pe').pct_change(1).over('tic').alias('pe_qoq'),
        pl.col('pe').pct_change(4).over('tic').alias('pe_yoy'),
        pl.col('ev_ebitda').pct_change(1).over('tic').alias('ev_eb_qoq'),
        pl.col('ev_ebitda').pct_change(4).over('tic').alias('ev_eb_yoy')
    ]).collect()
    return df
