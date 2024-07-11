import pandas as pd
import numpy as np
import datetime as dt

from config import get_config
from database_handler import DatabaseHandler

import talib as ta

YEARLY_TRADING_DAYS = 252
QUARTERLY_TRADING_DAYS = 60

# TODO: vectorize feature engineering


class Preprocess():
    """
    Data preprocessing handling class.
    """

    def __init__(self):
        self.db_handler = DatabaseHandler()
        self.features = get_config("data")["features"]
        self.targets = get_config("data")["targets"]

    def process_data(self):

        # read info on available stocks
        self.stock_info = self.db_handler.fetch_stock_info()
        
        # compute index forward returns 
        self.index_data = self.process_index_data()
        
        self.feature_engineering_vec()
            
    def process_index_data(self):
        
        # load index data
        index_data = self.db_handler.fetch_sp_data()
        
        # compute index forward returns 
        index_data['sp500_freturn'] = index_data['adj_close'].transform(
            lambda x: x.shift(-YEARLY_TRADING_DAYS) / x - 1
        )
        return index_data
    
    def feature_engineering_vec(self):
        """
        Compute financial ratios and features for training.
        """
        
        # get financial data
        df = self.db_handler.fetch_financial_data()
        df = df.sort_values(by=['tic', 'rdq'])
        df = df[df.tic.isin(['AAPL', 'AMZN'])]
        
        # set base date
        df['obs_date'] = df['rdq'] + pd.DateOffset(1)
        df['prev_rdq'] = df.groupby('tic')['rdq'].shift(periods=1)
        
        metadata = self.db_handler.fetch_metadata()
        market_df = self.db_handler.fetch_market_data()
        insider_df = self.db_handler.fetch_insider_data()
        
        df = df.groupby(['tic']).transform(compute_shares, metadata)

        # add each class of features
        df = compute_insider_trading_features(df, insider_df)
        df = compute_financial_ratios(df)
        df = compute_growth_ratios(df)
        df = compute_market_ratios(df, market_df)
        
        df = df[['obs_date', 'tic', 'sector'] + self.features + self.targets]
        
        return df
    
    def feature_engineering(self, stock_info):
        """
        Compute financial ratios and features for training.
        """
        
        # unpack info
        tic, sector = stock_info['tic'], stock_info['sector']
             
        # get financial data
        df = self.db_handler.fetch_financial_data(tic)
        
        if df.empty:
            raise Exception(f"No data for {tic}")
        
        # load from other sources
        metadata = self.db_handler.fetch_metadata(tic)
        market_df = self.db_handler.fetch_market_data(tic)
        insider_df = self.db_handler.fetch_insider_data(tic)
            
        # read current shares outstanding from metadata             
        curr_csho = float(metadata[1]) if metadata else float(df['cshoq'].iloc[-1]) * 1000000

        # set base date
        df['obs_date'] = df['rdq'] + pd.DateOffset(1)
        df['prev_rdq'] = df['rdq'].shift(periods=1)
        
        # add each class of features
        df = compute_insider_trading_features(df, insider_df)
        df = compute_financial_ratios(df)
        df = compute_growth_ratios(df)
        df = compute_market_ratios(df, market_df, curr_csho)
        
        df['tic'] = tic
        df['sector'] = sector
        df = df[['obs_date', 'tic', 'sector'] + self.features + self.targets]
        
        return df
    
    
def compute_shares(df, metadata):
    
    df['curr_csho'] = (float(metadata[1]) if metadata 
        else float(df['cshoq'].iloc[-1]) * 1000000)
    
    return df

def compute_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a selected number of financial ratios.

    Parameters
    ----------
    df : pd.DataFrame
        Main stock dataset.

    Returns
    -------
    pd.DataFrame
        Data with additional columns.
    """
    df['roa'] = df['niq'] / df['atq'].rolling(2).mean() # return on assets
    df['roe'] = df['niq'] / df['seqq'] # return on sh equity
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
    df['atr'] = df['saleq'] / df['atq'].rolling(2).mean()
    return df


def compute_growth_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling growth statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Main stock dataset.

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

def compute_market_ratios(df: pd.DataFrame, market_df: pd.DataFrame, csho: float) -> pd.DataFrame:
    

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
    
    # merge data
    df = pd.merge_asof(
        df, 
        market_df, 
        left_on='trade_date', 
        right_on='filling_date', 
        direction='forward',
        tolerance=dt.timedelta(days=7)
    )
    
    # compute p/e and eps
    df['eps'] = df['niq'] / csho
    df['pe'] = df['close'] / df['eps']
    
    # compute ev and ev to ebitda
    df['ev'] = df['close'] * csho + df['ltq'] - df['cheq']
    df['ev_ebitda'] = df['ev'] / df['ebitdaq']
    
    return df


def compute_insider_trading_features(df: pd.DataFrame, insider_df: pd.DataFrame) -> pd.DataFrame:

    df['key'] = 1
    insider_df['key'] = 1
    merged_df = pd.merge(df, insider_df, on='key').drop('key', axis=1)

    # filter the merged DataFrame
    count_p = merged_df[
        (merged_df['filling_date'] >= merged_df['prev_rdq']) & 
        (merged_df['filling_date'] <= merged_df['obs_date']) & 
        (merged_df['transaction_type'] == 'P - Purchase')
    ].groupby('obs_date').size().reset_index(name='n_purchases')
    
    # set nr of insider purchases in last quarter
    df['n_purchases'] = pd.merge(df, count_p, on='obs_date', how='left')['n_purchases'].fillna(0)

    return df
