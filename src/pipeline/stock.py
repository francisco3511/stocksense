import os
import datetime
import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path

import utils.market_scraper as ms
from config import get_config_dict


class Stock:
    """
    Class for stock asset representation.
    """

    def __init__(self, tic):
        self.tic = tic
        self.name = None
        self.sector = None
        self.subsector = None
        self.founded = None
        self.crsp_fields = get_config_dict("data")["crsp_columns"]
        self.base_date = get_config_dict("data")["base_date"]
        self.fin_last_update = None
        self.mkt_last_update = None
        self.data_path = None
        self._add_information()
        self._initialize_stock()

    def _add_information(self):
        """
        Add details and information on stock.
        """

        # read industry sector data (GICS terminology)
        control_df = pd.read_csv(
            './data/1_work_data/SP500.csv',
            sep=';',
        )
        
        try:
            stock_details = control_df.loc[
                control_df.Symbol == self.tic,
                ['Security', 'Sector', 'Sub-Sector', 'Founded']
            ].values
            
            self.name = stock_details[0, 0]
            self.sector = stock_details[0, 1]
            self.subsector = stock_details[0, 2]
            self.founded = stock_details[0, 3]
            
        except Exception:
            logger.error(f"{self.tic} add_info FAILED")

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
            self._retrieve_historical_insider_data()
            self.retrieve_insider_data(update=False)
            
        else:
            # gather date of last update
            fun_filename = list(self.data_path.glob('fundamentals_*.csv'))[0]
            mkt_filename = list(self.data_path.glob('market_*.csv'))[0]
            
            # store date of last update
            self.fin_last_update = datetime.datetime.strptime(fun_filename.stem.split('_')[1], '%Y-%m-%d').date()
            self.mkt_last_update = datetime.datetime.strptime(mkt_filename.stem.split('_')[1], '%Y-%m-%d').date()
            
            logger.info(f"repository for {self.tic} exists ({self.mkt_last_update})")
            
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
            self.fin_last_update = df['rdq'].dt.date.max()
            
            # save data
            df.to_csv(self.data_path / f'fundamentals_{self.fin_last_update}.csv', index=False)
            
            logger.info(f"retrieved crsp historical data for {self.tic} ({self.fin_last_update})")
        
        except Exception:
            
            # if no data was found create an empty df as placeholder
            df = pd.DataFrame(columns=df.columns)
            
            # reset date to default
            self.fin_last_update = datetime.datetime.strptime(self.base_date, '%Y-%m-%d').date()
            
            # save data 
            df.to_csv(self.data_path / f'fundamentals_{self.fin_last_update}.csv', index=False)
            
            logger.warning(f"no crsp historical data found for {self.tic}")
            
    def _retrieve_historical_market_data(self):
        
        try:
            # set end date to present
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            
            # scrape market data
            data = ms.get_market_data(
                self.tic, 
                start=self.base_date,
                end=end_date
            )
            
            # store date of last update
            self.mkt_last_update = data['Date'].dt.date.max()
            
            # save as csv
            data.to_csv(self.data_path / f'market_{self.mkt_last_update}.csv', index=False)
            logger.info(f"historical market data found on yahoo for {self.tic} ({self.mkt_last_update})")
            
        except Exception:
            # try on local DB 
            try:            
                # retrieve CRSP data
                market_data = pd.read_csv(
                    './data/0_raw_data/prices_2005-01-01_2018-12-31.csv', 
                    header=[0,1]
                ).loc[1:]
                
                # store single stock data 
                data = pd.DataFrame(columns=['Date', 'Close', 'Adj Close', 'Volume'])
                data['Date'] = market_data[market_data.columns[0]]
                data['Close'] = market_data['Close'][self.tic]
                data['Adj Close'] = market_data['Adj Close'][self.tic]
                data['Volume'] = market_data['Volume'][self.tic].astype(int, errors='ignore')
                
                # get last price date
                self.mkt_last_update = pd.to_datetime(data['Date']).dt.date.max()
                
                # save as csv
                data.to_csv(self.data_path / f'market_{self.mkt_last_update}.csv', index=False)
                logger.info(f"historical market data found on db for {self.tic} ({self.mkt_last_update})")
                
            except Exception:
                logger.warning(f"no market data available for {self.tic}")
                data = pd.DataFrame(columns=['Date', 'Close', 'Adj Close', 'Volume'])
                self.mkt_last_update = self.base_date
                data.to_csv(self.data_path / f'market_{self.base_date}.csv', index=False)
                        
    def update_fundamental_data(self, method):
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
            
            # extract date of last update and add 1 day
            start_dt = (datetime.datetime.combine(
                self.fin_last_update,
                datetime.datetime.min.time()
                ) + datetime.timedelta(days=1)
            )
        except Exception:
            # reset to default
            start_dt = datetime.datetime.strptime(self.base_date, '%Y-%m-%d')
            
        # set end date to present
        end_dt = datetime.datetime.now()
        
        # check if earnings season is possible
        if (end_dt - start_dt) < datetime.timedelta(days=75):
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
            
            # get earnings dates and estimates
            earn_dates = ms.get_earnings_dates(self.tic, start_dt, end_dt)
            
            # merge data
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
            self.fin_last_update = df['rdq'].dt.date.max()
            
            # save data
            df.to_csv(
                self.data_path /
                f'fundamentals_{self.fin_last_update}.csv',
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
        
    def update_market_data(self):
        """
        Updata market data (price and volume).
        TODO: scrape only missing data (from last update till present)
        and adjust past data if any split happened in meantime.

        Returns
        -------
        bool
            Signals success of update.

        Raises
        ------
        Exception
            No data is available.
        """
        
        # get current path
        mkt_filename = list(self.data_path.glob('market_*.csv'))[0]
        
        # extract date of last update
        last_update_date = datetime.datetime.strptime(
            mkt_filename.stem.split('_')[1], 
            '%Y-%m-%d'
        ).date()
        
        # set end date to present
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        
        try:
            if datetime.datetime.now().date() <= self.mkt_last_update:
                raise Exception("no market data to update.")
            
            # scrape market data
            data = ms.get_market_data(
                self.tic,
                self.base_date,
                end_date
            )
                        
            # save updated df
            data.to_csv(
                self.data_path /
                f'market_{end_date}.csv', 
                index=False
            )
            
            self.mkt_last_update = data['Date'].dt.date.max()
            
            # replace outdated data
            if os.path.exists(mkt_filename):
                os.remove(mkt_filename)
                
            logger.info(f"updated market data for {self.tic} ({self.mkt_last_update})")
            
            return True
        
        except Exception:
            logger.warning(f"no market data to update for {self.tic}")
            return False
        
    def retrieve_insider_data(self, update=False):
        """
        Get insider trading data.
        """
                      
        # set end date to present
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        
        try:
            # scrape insider data
            data = ms.get_stock_insider_data(self.tic)
            
            if update:
                
                # get historical insider trades
                insider_filename = list(self.data_path.glob('insider_*.csv'))[0]
                old_data = pd.read_csv(insider_filename)

                # join
                data = pd.concat([old_data, data]).reset_index(drop=True)
            
                # update
                data = data.drop_duplicates(subset=['filling_date', 'owner_name', 'transaction_type'], keep='last')
                
                # replace outdated data
                if os.path.exists(insider_filename):
                    os.remove(insider_filename)
                    
                logger.info(f"updated insider trading data for {self.tic} ({end_date})")
            else:
                logger.info(f"created insider trading data for {self.tic} ({end_date})")
                             
            # save updated df
            data.to_csv(
                self.data_path /
                f'insider_{end_date}.csv', 
                index=False
            )
            
            return True
        
        except Exception:
            logger.warning(f"no insider trading data found for {self.tic}")
            return False
            
    def update_stock_data(self, method='yh'):
        
        logger.info(f"updating {self.tic} stock data")
        
        # update financial data
        self.update_fundamental_data(method)
        
        # update market data
        self.update_market_data()
        
        # update insider trading data
        self.retrieve_insider_data(update=True)
          
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
    
    def get_insider_data(self):
        
        curr_data_path = list(self.data_path.glob('insider_*.csv'))[0]
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
        return self.mkt_last_update
    
    def get_ticker(self):
        return self.tic
