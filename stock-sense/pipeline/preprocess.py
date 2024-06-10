import pandas as pd
import numpy as np
from config import get_config_dict

class Preprocess():

    def __init__(self, financial_data, market_data, index_data):
        
        self.financial_data = financial_data
        self.market_data = market_data
        self.index_data = index_data
        self.features = get_config_dict("data")["variables"]
        
        
    def feature_engineering(self):
        
        fin_df = self.financial_data.copy()
        mkt_df = self.market_data.copy()
    
        #master = self._add_financial_ratios(fin_df)
        master = self._add_market_ratios(mkt_df)
        #master = self._add_growth_ratios(df)
        
        return master[['tic', 'datadate', 'saleq', 'invtq', 'rev_g4q', 'inv_avg']]
    
    
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