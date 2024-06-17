import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from pipeline import Stock
from utils import list_sp500_stocks, get_market_data


class Etl:
    """
    ETL handler for fundamental and price data.
    """

    def __init__(self, active=True):
        self._update_stock_listings()
        self.stock_list = self._set_stocks(active)
        self.stocks = []
        self.base_data_path = Path('./data/0_raw_data/')
        self.stock_data_path = Path('./data/1_work_data/company_data/')

    def _update_stock_listings(self):
        """
        Update S&P500 control file.
        """
        
        logger.info("getting S&P500 listings")

        # read control file
        hist_df = pd.read_csv(
            './data/1_work_data/SP500.csv',
            sep=';'
        )

        # read active sp500 stocks/sectors
        active_df = list_sp500_stocks()
        constituents_list = active_df['Symbol'].values.tolist()

        # join and update
        df = pd.concat([hist_df, active_df])
        df['Symbol'] = df['Symbol'].str.replace('.','-')
        df = df.drop_duplicates(subset=['Symbol'], keep='last').reset_index(drop=True)
        
        # flag current constituents
        df['Status'] = np.where(
            df['Symbol'].isin(constituents_list),
            True,
            False
        )

        # save results
        df.to_csv(
            './data/1_work_data/SP500.csv',
            sep=';',
            index=False
        )

    def _set_stocks(self, active=True) -> list[str]:
        """
        Set ETL stocks.
        """

        # read control file
        control_df = pd.read_csv(
            './data/1_work_data/SP500.csv',
            sep=';'
        )
        
        if active:
            # return current index constituents
            return control_df.loc[
                control_df['Status'] == True,  # noqa: E712
                'Symbol'
            ].values.tolist()
        else:
            # return all stocks (past / present)
            return control_df['Symbol'].values.tolist()

    def extract_sp_500(self):
        """
        Retrieve updated S&P500 data.
        """

        logger.info("extracting S&P500 data")

        # set update date to present
        update_date = datetime.datetime.now().strftime('%Y-%m-%d')
        base_date = '2005-01-01'

        # scrape index data
        data = get_market_data('^GSPC', base_date, update_date)

        # save data
        data.to_csv(
            self.base_data_path /
            'SP500_price.csv',
            index=False
        )

    def extract_stock_data(self):
        """
        Extract stock all stock data from pre-defined data sources.

        Args:
            active (bool, optional): Set to only extract active
            S&P500 constituents. Defaults to True.
        """

        logger.info("Start stock data extraction process")

        pl_bar = tqdm(total=len(self.stock_list), desc='Stock', leave=True)
        
        for tic in self.stock_list:
            logger.info(f"retrieving {tic} data")

            # create stock instance
            stock = Stock(tic)

            # update stock data
            #stock.update_stock_data(method='yh')
            stock.retrieve_insider_data(update=False)
            
            # add stock instance
            self.stocks.append(stock)
            
            # update progress
            pl_bar.update(1)
            
        pl_bar.close()

    def load_all_data(self) -> dict:

        logger.info(f"loading all stock data")
        
        data = {}

        for stock in self.stocks:
            # append stock data
            data[stock.get_ticker()]["market"] = stock.get_market_data()
            data[stock.get_ticker()]["financial"] = stock.get_financial_data()
            data[stock.get_ticker()]["insider"] = stock.get_insider_data()
            
        data['SPX'] = self.load_index_data()
        
        return data
    
    def load_stock_data(self, ticker) -> tuple[pd.DataFrame]:

        logger.info(f"loading {ticker} stock data")
        
        for stock in self.stocks:
            if stock.get_ticker() == ticker:
                return (
                    stock.get_financial_data(),
                    stock.get_market_data(),
                    stock.get_insider_data()
                )

    def load_index_data(self):
        """
        Load S&P500 price data.

        Returns
        -------
        pd.DataFrame
            S&P500 price data.
        """

        curr_data_path = list(self.base_data_path.glob('SP500_price.csv'))[0]
        df = pd.read_csv(
            curr_data_path,
            parse_dates=['Date'],
            date_format='%Y-%m-%d'
        )
        return df
