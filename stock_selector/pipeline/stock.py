import os
import datetime
from loguru import logger
from pathlib import Path
import numpy as np
import pandas as pd

import utils.market_scraper as ms
from config import get_config_dict


class Stock:
    """
    Class for stock asset representation.
    """

    def __init__(self, tic):
        self.tic = tic
        self.sector = None
        self.crsp_fields = get_config_dict("data")["crsp_columns"]
        self.base_date = get_config_dict("data")["base_date"]
        self.update_date = None
        self.data_path = None
        self._add_sector()
        self._initialize_stock()

    def _add_sector(self):
        """
        Add sector information to stock, from persistent database.
        """

        # Read industry sector data (GICS terminology)
        sector_data = pd.read_csv(
            './data/1_work_data/SP500.csv',
            sep=';',
            usecols=['Symbol', 'Sector']
        )
        try:
            self.sector = sector_data.loc[
                sector_data.Symbol == self.tic,
                'Sector'
            ].values[0]
        except Exception:
            logger.error(f"{self.tic} add_sector FAILED")
            self.sector = None

    def _initialize_stock(self):
        """
        Initialize stock repository by:
         - creating a folder
         - checking for historical financial data on CRSP database
         - scraping market data.
        If stock exists, simply gather date of last update.
        """

        # set stock data storage path 
        self.data_path = Path(f'./data/1_work_data/company_data/{self.tic}')
        
        # check if path exists 
        if not os.path.exists(self.data_path):
            
            logger.info(f"creating repository for {self.tic} ...")
            
            # create dir for stock
            os.makedirs(self.data_path)
            
            # retrieve and save historical financial data and store last update date
            self._retrieve_historical_fundamental_data()
            self._retrieve_historical_market_data()
            
        else:
            # gather date of last update
            curr_f_data_name = list(self.data_path.glob('fundamentals_*.csv'))[0]
            date = datetime.datetime.strptime(curr_f_data_name.stem.split('_')[1], '%Y-%m-%d').date()
            
            # store date of last update
            self.update_date = date
            logger.info(f"repository for {self.tic} exists ({self.update_date})")
            
    def _retrieve_historical_fundamental_data(self):
        """
        Read CRSP database snapshot and retrieve stock data, parsing the financial fields 
        passed on config file. 
        """

        try:
            # retrieve CRSP data
            crsp_data = pd.read_csv(
                './data/0_raw_data/crsp/fundamentals_quarters_2005-2018.csv', 
                usecols=self.crsp_fields
            )
            
            # retrieve firm data
            df = crsp_data.loc[crsp_data.tic == self.tic]
                
            # clear CRSP duplicates
            df = df[(df.indfmt == 'INDL') & (df.datafmt == 'STD')]
                
            # parse date of record into datetime format
            df['datadate'] = pd.to_datetime(df['datadate'])
                
            # releasedate of fin statements
            df['rdq'] = pd.to_datetime(df['rdq'], dayfirst=True)
            df['rdq'].fillna(df['datadate'] + pd.Timedelta(days=60), inplace=True)
            df.sort_values(by=['rdq'], inplace=True)
                
            # quarterly adjustments (for cumulative annual fields)
            df['oancfq'] = df.groupby(['fyearq', 'tic'])['oancfy'].diff().fillna(df['oancfy'])
            df['ivncfq'] = df.groupby(['fyearq','tic'])['ivncfy'].diff().fillna(df['ivncfy'])
            df['fincfq'] = df.groupby(['fyearq','tic'])['fincfy'].diff().fillna(df['fincfy'])
            df['capxq'] = df.groupby(['fyearq', 'tic'])['capxy'].diff().fillna(df['capxy'])
            
            # generate quarterly dividend payouts (total) where DVY = Cash Dividends (annual field)
            df['dvq'] = df.groupby(['fyearq', 'tic'])['dvy'].diff().fillna(df['dvy'])        
            
            # add ebitda
            df['ebitdaq'] = df['saleq'] - df['cogsq'] - df['xsgaq']
            df['ebitdaq'].fillna(df['niq'] + df['txtq'] + df['xintq'] + df['dpq'], inplace=True)
            
            # add placeholder for eps est, reported eps and surprise %
            df['eps_est'] = np.nan
            df['eps_rep'] = np.nan
            df['surprise_pct'] = np.nan
            
            # adds sector
            df['sector'] = self.sector
            
            # drop unnecessary fields
            df.drop(
                columns=[
                    'oancfy', 'ivncfy', 'fincfy', 'capxy',
                    'gvkey', 'datafmt', 'indfmt', 'fyearq', 
                    'fqtr', 'txtq', 'xintq', 'dpq', 'dvy'],
                inplace=True
            )

            if not len(df):
                raise Exception("Empty DataFrame")
            
            # store date of last update
            self.update_date = df['rdq'].dt.date.max()
            df.to_csv(self.data_path / f'fundamentals_{self.update_date}.csv', index=False)
            logger.info(f"retrieved crsp historical data for {self.tic} ({self.update_date})")
        
        except Exception:
            # if no data was found create an empty df as placeholder
            df = pd.DataFrame(columns=df.columns)
            # reset date to default
            self.update_date = datetime.datetime.strptime(self.base_date, '%Y-%m-%d').date()
            df.to_csv(self.data_path / f'fundamentals_{self.update_date}.csv', index=False)
            logger.warning(f"no crsp historical data found for {self.tic}")
            
    def _retrieve_historical_market_data(self):
        """
        Retrieve market data from 
        """
        
        try:
            # set end date to present
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            # scrape market data
            data = ms.get_market_data(
                self.tic, 
                start=self.base_date,
                end=end_date
            )
            # save as csv
            data.to_csv(self.data_path / f'market_{end_date}.csv', index=False)
            logger.info(f"historical market data found on yahoo for {self.tic} ({end_date})")
            
        except Exception:
            # try on local DB 
            try:            
                # retrieve CRSP data
                market_data = pd.read_csv('./data/0_raw_data/prices_2005-01-01_2018-12-31.csv', 
                                        header=[0,1]).loc[1:]
                
                # store single stock data 
                data = pd.DataFrame(columns=['Date', 'Close', 'Adj Close', 'Volume'])
                data['Date'] = market_data[market_data.columns[0]]
                data['Close'] = market_data['Close'][self.tic]
                data['Adj Close'] = market_data['Adj Close'][self.tic]
                data['Volume'] = market_data['Volume'][self.tic].astype(int, errors='ignore')
                
                # get last price date
                update_date = pd.to_datetime(data['Date']).dt.date.max()
                
                # save as csv
                data.to_csv(self.data_path / f'market_{update_date}.csv', index=False)
                logger.info(f"historical market data found on db for {self.tic} ({update_date})")
                
            except Exception:
                logger.warning(f"no market data available for {self.tic}")
                data = pd.DataFrame(columns=['Date', 'Close', 'Adj Close', 'Volume'])
                data.to_csv(self.data_path / f'market_{self.base_date}.csv', index=False)
        
    def _update_fundamental_data(self, method):
        """
        Update fundamental data, by retrieving financial reports released
        after previous updates.

        Args:
            method (str, optional): Data sources, either yh (yahoo) or macrotrends (mt).

        Returns:
            bool: Success status.
        """
        
        try:
            # get current path
            curr_data_name = list(self.data_path.glob('fundamentals_*.csv'))[0]
            # extract date of last update, convert to dt and add 1 day
            start_dt = (datetime.datetime.combine(
                self.update_date,
                datetime.datetime.min.time()
                ) + datetime.timedelta(days=1)
            )
        except Exception:
            # reset to default
            start_dt = datetime.datetime.strptime(self.base_date, '%Y-%m-%d')
            
        # set end date to present
        end_dt = datetime.datetime.now()
        
        # check if earnings season is possible
        if end_dt - start_dt < datetime.timedelta(days=75):
            logger.info(f'financial data already updated for {self.tic} ({end_dt})')
            return False
        
        try:
            # update via macrotrends
            if method == 'mt': 
                # scrape and merge 3 main documents
                df_is =  ms.scrape_income_statement_mt(self.tic, start_dt, end_dt)
                df_bs =  ms.scrape_balance_sheet_mt(self.tic, start_dt, end_dt)
                df_cfs =  ms.scrape_cash_flow_mt(self.tic, start_dt, end_dt)
                df_ratios = ms.scrape_ratios_mt(self.tic, start_dt, end_dt)
                
                df = pd.merge(df_is, df_bs, left_on='datadate', right_on='datadate', how='inner')
                df = pd.merge(df, df_cfs, left_on='datadate', right_on='datadate', how='inner')
                df = pd.merge(df, df_ratios, left_on='datadate', right_on='datadate', how='inner')
                
                # corrections
                df['icaptq'] = (df['niq'] / df['icaptq']) * 100
                df['dvq'] = -df['dvq']
            
            # update via yahoo finance
            elif method == 'yh':
                # scrape fundamental data from yahoo
                df = ms.scrape_fundamental_data_yahoo(self.tic, start_dt, end_dt)
            
            # add missing columns
            df['tic'] = self.tic
            df['sector'] = self.sector
            
            # corrections
            df = df.drop_duplicates(subset=['datadate']).sort_values(by=['datadate'])
            df.sort_values(by=['datadate'], inplace=True)
            
            # get earnings dates
            earn_dates = ms.get_earnings_dates(self.tic, start_dt, end_dt)
            df = pd.merge_asof(
                df, 
                earn_dates, 
                left_on='datadate', 
                right_on='rdq', 
                direction='forward',
                tolerance=datetime.timedelta(days=80)
            )
            # retrieve outdated fundamental data
            old_data = pd.read_csv(curr_data_name)
            df = pd.concat([old_data, df]).reset_index(drop=True)
            
            # update
            df = df.drop_duplicates(subset=['datadate'], keep='last')
            
            # parse dates
            df['datadate'] = pd.to_datetime(df['datadate'])    
            df['rdq'] = pd.to_datetime(df['rdq'])
            
            # round data
            df = df.round(3)
                        
            # set update date
            self.update_date = df['rdq'].dt.date.max()
            df.to_csv(
                self.data_path /
                f'fundamentals_{self.update_date}.csv',
                index=False
            )
            # remove outdated data
            if os.path.exists(curr_data_name):
                os.remove(curr_data_name)
                
            logger.info(f'updated financial data for {self.tic} ({start_dt} : {end_dt})')
            return True
        except Exception:
            logger.warning(f'failed to update financial data ({method}) for {self.tic} ({end_dt})')
            return False
        
    def _update_market_data(self):
        
        # get current path
        curr_data_name = list(self.data_path.glob('market_*.csv'))[0]
        
        # extract date of last update
        last_update_date = datetime.datetime.strptime(
            curr_data_name.stem.split('_')[1], 
            '%Y-%m-%d'
        ).date()
        
        # set end date to present
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        
        try:
            if last_update_date > self.update_date:
                raise Exception("no market data to update.")
            
            # scrape market data
            data = ms.get_market_data(self.tic, self.base_date, end_date)
                        
            # save updated df
            data.to_csv(
                self.data_path /
                f'market_{end_date}.csv', 
                index=False
            )
            
            # replace outdated data
            if os.path.exists(curr_data_name):
                os.remove(curr_data_name)
                
            logger.info(f"updated market data for {self.tic} ({end_date})")
            
            return True
        
        except Exception:
            logger.warning(f"no market data to update for {self.tic}")
            
            return False
            
    def update_stock_data(self, method='yh'):
        
        logger.info(f"updating {self.tic} stock data")
        
        # update financial data
        self._update_fundamental_data(method)
        
        # update market data
        self._update_market_data()
          
    def correct_data(self):
        
        curr_data_name = list(self.data_path.glob('fundamentals_*.csv'))[0]
        
        df = pd.read_csv(curr_data_name)
        
        # parse dates
        df['datadate'] = pd.to_datetime(df['datadate'])    
        #df = df.drop(columns=['rdq', 'epsfiq', 'epsfi12', 'epspiq'])
        
        cols_to_retain = ['datadate', 'rdq', 'tic'] + list(df.columns)[3:9] + \
            list(df.columns)[12:-1] + ['epspiq', 'epsfiq', 'epsfi12', 'eps_est', 'eps_rep', 'surprise_pct', 'sector']
        
        try:
            earn_data = ms.get_earnings_dates(self.tic, n_historical=100)
            df = df.drop(columns=['rdq'])
            df = pd.merge_asof(
                df,
                earn_data, 
                left_on='datadate', 
                right_on='rdq', 
                direction='forward', 
                tolerance=datetime.timedelta(days=75)
            )
            df['rdq'].fillna(df['datadate'] + pd.Timedelta(days=60), inplace=True)
            df = df[cols_to_retain]
            df = df.round(3) 
            df.to_csv(curr_data_name, index=False)
            logger.info(f'corrected ({self.tic})')
            return self.tic
        except Exception:
            try: 
                df['eps_est'] = np.nan
                df['eps_rep'] = np.nan
                df['surprise_pct'] = np.nan
                df = df[cols_to_retain]
                df = df.round(3)
                df.to_csv(curr_data_name, index=False)
                logger.info(f'std correction ({self.tic})')
                return False
            except Exception:
                logger.warning(f'No data corrections ({self.tic})')
                return False
            
    def get_financial_data(self):
        
        curr_data_path = list(self.data_path.glob('fundamentals_*.csv'))[0]
        df = pd.read_csv(
            curr_data_path, 
            parse_dates=['datadate', 'rdq'], 
            date_format='%Y-%m-%d'
        )
        return df
    
    def get_market_data(self):
        
        curr_data_path = list(self.data_path.glob('market_*.csv'))[0]
        df = pd.read_csv(
            curr_data_path, 
            parse_dates=['Date'], 
            date_format='%Y-%m-%d'
        )    
        return df
    
    def is_empty(self):
        
        if (not self.get_data('f')) or (not self.get_data('m')):
            return True
        else:
            return False
    
    def get_updated_date(self):
        return self.update_date
