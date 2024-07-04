import pandas as pd
import numpy as np
import datetime as dt
from config import get_config

import talib as ta

YEARLY_TRADING_DAYS = 252
QUARTERLY_TRADING_DAYS = 60


class Preprocess():

    def __init__(self, data):
        self.index_data = data.pop("SP500")
        self.stock_data = data
        self.features = get_config("data")["variables"]
        
    def feature_engineering(self):
        """
        Compute financial ratios and features for training.
        """
                
        # compute index forward returns 
        self.index_data['sp500_forward_return'] = self.index_data['Adj Close'].transform(
            lambda x: x.shift(-YEARLY_TRADING_DAYS) / x - 1
        )
        
        # create pandas instance for dataset
        dataset = pd.DataFrame(
            columns=self.features
        )
        
        values = []
        
        for tic, data in self.stock_data.items():
            # unpack data
            df = data[0]
            info_df = data[1]
            mkt_df = data[2]
            ins_df = data[3]
    
            
            if not len(mkt_df) or not len(df):
                continue
            
            if 'sharesOutstanding' not in info_df:
                csho = float(df['cshoq'].iloc[-1]) * 1000000
            else:
                csho = float(info_df['sharesOutstanding'])
                
            # set base date
            df['obs_date'] = df['rdq'] + pd.DateOffset(1)
            df['prev_rdq'] = df['rdq'].shift(periods=1)
            
            # compute insider trading features
            df['key'] = 1
            ins_df['key'] = 1
            merged_df = pd.merge(df, ins_df, on='key').drop('key', axis=1)

            # Filter the merged DataFrame
            count_p = merged_df[
                (merged_df['filling_date'] >= merged_df['prev_rdq']) & 
                (merged_df['filling_date'] <= merged_df['obs_date']) & 
                (merged_df['transaction_type'] == 'P - Purchase')
            ].groupby('obs_date').size().reset_index(name='n_purchases')
            

            df['n_purchases'] = pd.merge(df, count_p, on='obs_date', how='left')['n_purchases'].fillna(0)
            
            # compute financial ratios
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

            # growth ratios
            df['rev_yoy'] = (df['saleq'] - df['saleq'].shift(periods=4)) / df['saleq'].shift(periods=4)
            df['ni_qoq'] = (df['niq'] - df['niq'].shift(periods=1)) / df['niq'].shift(periods=1)
            df['ni_yoy'] = (df['niq'] - df['niq'].shift(periods=4)) / df['niq'].shift(periods=4)
            df['ni_2y'] = (df['niq'] - df['niq'].shift(periods=8)) / df['niq'].shift(periods=8)
            df['ltd_yoy'] = (df['dlttq'] - df['dlttq'].shift(periods=4)) / np.abs(df['dlttq'].shift(periods=4))
            df['dr_yoy'] = (df['dr'] - df['dr'].shift(periods=4)) / np.abs(df['dr'].shift(periods=4))
    
             
            # compute forward returns 
            mkt_df[f'forward_return'] = mkt_df['Adj Close'].transform(
                    lambda x: x.shift(-YEARLY_TRADING_DAYS) / x - 1
            )
            # compute momentum indicators
            mkt_df['rsi_9d'] = ta.RSI(mkt_df['Close'], timeperiod=9)
            mkt_df['rsi_30d'] = ta.RSI(mkt_df['Close'], timeperiod=30)
            
            # price growth ratios
            mkt_df['price_yoy'] = (
                (mkt_df['Close'] - mkt_df['Close'].shift(periods=YEARLY_TRADING_DAYS)) /
                mkt_df['Close'].shift(periods=YEARLY_TRADING_DAYS)
            )
            mkt_df['price_qoq'] = (
                (mkt_df['Close'] - mkt_df['Close'].shift(periods=QUARTERLY_TRADING_DAYS)) /
                mkt_df['Close'].shift(periods=QUARTERLY_TRADING_DAYS)
            )
            
            df = pd.merge_asof(
                df, 
                mkt_df, 
                left_on='trade_date', 
                right_on='filling_date', 
                direction='forward',
                tolerance=dt.timedelta(days=7)
            )
            
            
            
            

            

        
        
        return 
    
    def _add_financial_ratios(self, df):
        
        # Gross Profit and Gross Profit Margin
        df['gpmq'] = ((df['saleq'] - df['cogsq']) / df['saleq']) * 100
        # Free Cash Flow
        df['fcfq'] = df['oancfq'] - df['capxq']
        # ROA
        df['roa'] = (df['niq'] / df['atq']) * 100
        # ROE
        df['roeq'] = (df['niq'] / df['seqq']) * 100
        # EBITDA Margin
        df['ebitdam'] = (df['ebitdaq'] / df['saleq']) * 100
        # Cash Flow Margin
        df['cfm'] = (df['oancfq'] / df['saleq']) * 100
        # Current Ratio
        df['cr'] = (df['actq'] / df['lctq']) * 100
        # Quick Ratio
        df['qr'] = ((df['rectq'] + df['cheq']) / df['lctq']) * 100
        # Cash Ratio
        df['csr'] = (df['cheq'] / df['lctq']) * 100
        # Sloan Ratio
        df['sloan'] = np.abs((df['niq'] - df['oancfq'] - df['ivncfq']) / df['atq'])
        # Debt Ratio
        df['dr'] = (df['ltq'] / df['atq']) * 100
        # Debt-to-Equity Ratio
        df['der'] = (df['ltq'] / df['seqq']) * 100
        # Debt to EBITDA
        df['debitda'] = (df['ltq'] / df['ebitdaq']) * 100
        # Long-Term Debt-to-Total-Assets
        df['ltda'] = (df['dlttq'] / df['atq']) * 100
        # Long Term Debt Coverage Ratio
        df['ltcr'] = (df['fcfq'] / df['dlttq']) * 100
        # Inventory Turnover Ratio
        df['inv_avg'] = df.groupby('tic')['invtq'].rolling(2).mean().reset_index(0,drop=True)
        df['invx'] = df['saleq'] / df.groupby('tic')['invtq'].rolling(2).mean().reset_index(0,drop=True)
        # Receivables Turnover Ratio
        #df['rectrq'] = df['saleq'] / df.groupby('tic')['rectq'].rolling(2).mean()
        # Asset Turnover Ratio
        #df['attq'] = df['saleq'] / df.groupby('tic')['atq'].rolling(2).mean()
        
        return df

    def _add_growth_ratios(self, df):
        
        # Sales growth
        df['rev_g4q'] = df.groupby('tic')['saleq'].pct_change(4) * 100
        # Net income growth
        df['ni_g4q'] = df.groupby('tic')['niq'].pct_change(4) * 100
        
        return df
    
    
    def _add_market_ratios(self, df):
        df = get_rsi(df, 14)
        return df
    
    
def get_rsi(close, lookback):
    ret = close.diff()
    up = []
    down = []
    for i in range(len(ret)):
        if ret[i] < 0:
            up.append(0)
            down.append(ret[i])
        else:
            up.append(ret[i])
            down.append(0)
    
    up_series = pd.Series(up)
    down_series = pd.Series(down).abs()
    up_ewm = up_series.ewm(com = lookback - 1, adjust = False).mean()
    down_ewm = down_series.ewm(com = lookback - 1, adjust = False).mean()
    rs = up_ewm/down_ewm
    rsi = 100 - (100 / (1 + rs))
    rsi_df = pd.DataFrame(rsi).rename(columns = {0:'rsi'}).set_index(close.index)
    rsi_df = rsi_df.dropna()
    return rsi_df[3:]