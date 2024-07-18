import pandas as pd
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
        
    def process_index_data(self) -> pd.DataFrame:
        """
        Processes index price data.

        Returns
        -------
        pd.DataFrame
            Daily index price data with computed features.
        """
        
        # load index data
        index_data = self.db_handler.fetch_sp_data()
        
        # parse date
        index_data['date'] = pd.to_datetime(
            index_data['date'],
            format='ISO8601',
            errors='coerce'
        )
        
        # compute index forward returns 
        index_data['index_freturn'] = index_data['adj_close'].transform(
            lambda x: x.shift(-YEARLY_TRADING_DAYS) / x - 1
        )
        return index_data

    def feature_engineering(self) -> pd.DataFrame:
        """
        Compute financial ratios and features for training.
        """
        
        # get financial data
        df = self.db_handler.fetch_financial_data()
        #df = df[df.tic.isin(['AAPL', 'AMZN'])]
        
        df['prev_rdq'] = df.groupby('tic')['rdq'].shift(periods=1)
        
        # convert dates to datetime and sort
        df['rdq'] = pd.to_datetime(
            df['rdq'],
            format='ISO8601',
            errors='coerce'
        )
        df['prev_rdq'] = pd.to_datetime(
            df['prev_rdq'],
            format='ISO8601',
            errors='coerce'
        )
        df = df.sort_values(by=['tic', 'rdq'])
        
        # fetch ALL other stock data from source tables
        info = self.db_handler.fetch_stock_info()
        market_df = self.db_handler.fetch_market_data()
        insider_df = self.db_handler.fetch_insider_data()
        
        # convert and sort market data
        market_df['date'] = pd.to_datetime(
            market_df['date'],
            format='ISO8601',
            errors='coerce'
        )
        market_df = market_df.sort_values(by=['tic', 'date'])
        
        # compute curr shares outstanding
        df = df.groupby('tic').apply(
            lambda x: compute_shares(x)
        ).reset_index(drop=True)

        # compute insider trading counts
        df = df.groupby('tic').apply(
            lambda x: compute_insider_trading_features(x, x.name, insider_df)
        ).reset_index(drop=True)

        # compute financial ratios
        df = df.groupby('tic').apply(
            lambda x: compute_financial_ratios(x)
        ).reset_index(drop=True)
        
        # compute growth metrics
        df = df.groupby('tic').apply(
            lambda x: compute_growth_ratios(x)
        ).reset_index(drop=True)
        
        # compute price/market ratios
        df = df.groupby('tic').apply(
            lambda x: compute_market_ratios(x, x.name, market_df, self.index_data)
        ).reset_index(drop=True)
        
        # add id info and select relevant features
        df = pd.merge(df, info[['tic', 'sector']], how='left', on='tic')
        
        df = df[
            ['obs_date', 'tic', 'sector', 'close']
            + self.features + self.targets
        ]
        return df
    
    def save_data(self, name: str) -> None:
        self.data.to_csv(f"data/2_production_data/{name}.csv")
    
def compute_shares(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes adjusted common outstanding shares,
    for historical analysis, based on stock-split detection.

    Parameters
    ----------
    df : pd.DataFrame
        Financial data of a given stock.

    Returns
    -------
    pd.DataFrame
        Data with adjusted outstd. shares.
    """
    
    # fill nan values
    df['cshoq'] = df['cshoq'].bfill()

    # compute the percentage change in common shares outstanding
    df['pct_change'] = df['cshoq'].pct_change(fill_method=None)

    # flag potential stock splits based on threshold
    df['stock_split'] = df['pct_change'].abs() > 0.25

    # initialize the adjusted shares column with the original values
    df['adj_shares'] = df['cshoq']
    
    common_split_ratios = [
        0.5, 0.33, 0.25, 
        2, 3, 4, 5, 6, 7, 8, 10, 20, 30
    ]

    for i, row in df.iterrows():
        if df.loc[i, 'stock_split']:
            # compute the raw split ratio 
            split_ratio = df.loc[i, 'cshoq'] / df.loc[i-1, 'cshoq']
            
            # match to common split ratios
            closest_split = min(common_split_ratios, key=lambda x:abs(x-split_ratio))
            if abs(closest_split - split_ratio) < 0.1:
                split_ratio = float(closest_split)
                
            # apply the split ratio to all previous shares
            df.loc[:i-1, 'adj_shares'] *= split_ratio
    
    return df.drop(['stock_split', 'pct_change'], axis=1)


def compute_insider_trading_features(
    df: pd.DataFrame,
    tic: str,
    insider_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes insider trading features
    (i.e. number of insider purchases throughout past quarter)

    Parameters
    ----------
    df : pd.DataFrame
        Financial data of a given stock.
    tic : str
        Stock ticker.
    insider_df : pd.DataFrame
        Global insider trading data.

    Returns
    -------
    pd.DataFrame
        Data with insider trading features.
    """
    
    try:
        # slice insider data
        insider_df = insider_df[insider_df.tic == tic]

        df['key'] = 1
        insider_df['key'] = 1
        merged_df = pd.merge(df, insider_df, on='key').drop('key', axis=1)

        # filter the merged DataFrame
        count_p = merged_df[
            (merged_df['filling_date'] >= merged_df['prev_rdq']) & 
            (merged_df['filling_date'] < merged_df['rdq']) & 
            (merged_df['transaction_type'] == 'P - Purchase')
        ].groupby('rdq').size().reset_index(name='n_purchases')
        
        # set nr of insider purchases in last quarter
        df = pd.merge(df, count_p, on='rdq', how='left')
        df['n_purchases'] = df['n_purchases'].fillna(0)
    except Exception:
        df['n_purchases'] = np.nan
    return df


def compute_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a selected number of financial ratios.

    Parameters
    ----------
    df : pd.DataFrame
        Financial data of a given stock.

    Returns
    -------
    pd.DataFrame
        Data with additional columns.
    """
    df['roa'] = df['niq'].rolling(4).sum() / df['atq'].rolling(2).mean()
    df['roe'] = df['niq'].rolling(4).sum() / df['seqq'] # return on sh equity
    df['gpm'] = (df['saleq'] - df['cogsq']) / df['saleq'] # gross profit margin
    df['ebitdam'] = df['ebitdaq'] / df['saleq'] # ebitda margin
    df['cfm'] = df['oancfq'] / df['saleq'] # cash flow margin
    df['cr'] = df['actq'] / df['lctq'] # current ratio
    df['qr'] = (df['rectq'] + df['cheq']) / df['lctq'] # quick ratio
    df['csr'] = df['cheq'] / df['lctq'] # cash ratio
    df['dr'] = df['ltq'] / df['atq'] # debt ratio
    df['der'] = df['ltq'] / df['seqq'] # debt-to-Equity ratio
    df['debitda'] = df['ltq'] / df['ebitdaq'] # debt to ebitda
    df['ltda'] = df['dlttq'] / df['atq'] # long term debt to assets
    df['ltcr'] = (df['oancfq'] - df['capxq']) / df['dlttq'] # long term debt coverage
    df['itr'] = df['saleq'] / df['invtq'].rolling(2).mean() # inventory turnover ratio
    df['rtr'] = df['saleq'] / df['rectq'].rolling(2).mean() # receivables turnover ratio
    df['atr'] = df['saleq'] / df['atq'].rolling(2).mean() # asset turnover ratio
    return df


def compute_growth_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling growth statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Financial data of a given stock.

    Returns
    -------
    pd.DataFrame
        Data with additional columns.
    """
    
    df['ni_qoq'] = (df['niq'] - df['niq'].shift(periods=1)) / df['niq'].shift(periods=1)
    df['ni_yoy'] = (df['niq'] - df['niq'].shift(periods=4)) / df['niq'].shift(periods=4)
    df['ni_2y'] = (df['niq'] - df['niq'].shift(periods=8)) / df['niq'].shift(periods=8)
    df['rev_yoy'] = (df['saleq'] - df['saleq'].shift(periods=4)) / df['saleq'].shift(periods=4)
    df['ltd_yoy'] = (df['dlttq'] - df['dlttq'].shift(periods=4)) / np.abs(df['dlttq'].shift(periods=4))
    df['dr_yoy'] = (df['dr'] - df['dr'].shift(periods=4)) / np.abs(df['dr'].shift(periods=4))
    return df

def compute_market_ratios(
    df: pd.DataFrame,
    tic: str,
    market_df: pd.DataFrame,
    index_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute market related ratios.

    Parameters
    ----------
    df : pd.DataFrame 
        Financial data of a given stock.
    tic : str
        Stock ticker.
    market_df : pd.DataFrame
        Market data.
    index_df : pd.DataFrame
        Index price data.

    Returns
    -------
    pd.DataFrame
        Main dataset with added ratios.
    """
    
    # slice insider data
    market_df = market_df[market_df.tic == tic]

    # compute forward returns 
    market_df['freturn'] = market_df['adj_close'].transform(
        lambda x: x.shift(-YEARLY_TRADING_DAYS) / x - 1
    )
    # compute momentum indicators
    market_df['rsi_9d'] = ta.RSI(market_df['close'], timeperiod=9)
    market_df['rsi_30d'] = ta.RSI(market_df['close'], timeperiod=30)
            
    # price growth ratios
    market_df['price_yoy'] = (
        (market_df['close'] - market_df['close'].shift(periods=YEARLY_TRADING_DAYS)) /
        market_df['close'].shift(periods=YEARLY_TRADING_DAYS)
    )
    market_df['price_qoq'] = (
        (market_df['close'] - market_df['close'].shift(periods=QUARTERLY_TRADING_DAYS)) /
        market_df['close'].shift(periods=QUARTERLY_TRADING_DAYS)
    )

    # merge market data
    df = pd.merge_asof(
        df, 
        market_df.drop(['tic', 'volume'], axis=1), 
        left_on='rdq', 
        right_on='date', 
        direction='forward',
        tolerance=dt.timedelta(days=7)
    )
    
    # merge index data
    df = pd.merge_asof(
        df, 
        index_df[['date', 'index_freturn']], 
        left_on='rdq', 
        right_on='date', 
        direction='forward',
        tolerance=dt.timedelta(days=7)
    )
    
    # set observation date
    df['obs_date'] = df['date_x']
    
    # adj forward returns by index returns
    df['adj_freturn'] = df['freturn'] - df['index_freturn']
    df['adj_fperf'] = (df['adj_freturn'] > 0.0).astype(int)
    df['adj_fperf_20'] = (df['adj_freturn'] > 0.2).astype(int)
    
    # compute eps and p/e
    df['eps'] = df['niq'].rolling(4).sum() / df['adj_shares']
    df['pe'] = df['close'] / df['eps']
    
    # compute ev to ebitda
    df['mkt_cap'] = df['close'] * df['adj_shares']
    df['ev'] = df['mkt_cap'] + df['ltq'] - df['cheq']
    df['ev_ebitda'] = df['ev'] / df['ebitdaq'].rolling(4).sum() 
    return df

