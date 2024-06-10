import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from pipeline.stock import Stock
from utils import list_sp500_stocks, get_market_data


class Etl:
    """
    ETL handler for fundamental and price data.
    """

    def __init__(self):
        self._update_stock_listings()
        self.stocks = self._set_base_stocks()
        self.active_stocks = self._set_active_stocks()
        self.base_data_path = Path('./data/0_raw_data/')
        self.stock_data_path = Path('./data/1_work_data/company_data/')

    def _update_stock_listings(self):
        """
        Update S&P500 control file.
        """

        # read control file
        hist_df = pd.read_csv(
            './data/1_work_data/SP500.csv',
            sep=';',
            usecols=['Symbol', 'Sector']
        )

        # read active sp500 stocks/sectors
        active_df = list_sp500_stocks()

        # join and update
        df = pd.concat([hist_df, active_df])
        df = df.drop_duplicates(subset=['Symbol']).reset_index(drop=True)
        active_stock_list = active_df['Symbol'].values.tolist()
        df['Status'] = np.where(
            df['Symbol'].isin(active_stock_list),
            True,
            False
        )

        # save results
        df.to_csv(
            './data/1_work_data/SP500.csv',
            sep=';'
        )

    def _set_base_stocks(self) -> list:
        """
        Set stocks for extraction.

        Returns:
            list: list of base stock tickers, including historical.
        """

        control_df = pd.read_csv(
            './data/1_work_data/SP500.csv',
            sep=';'
        )
        return control_df['Symbol'].values.tolist()

    def _set_active_stocks(self):
        """
        Set active stocks, based on stocks currently
        listed on the S&P500 index.

        Returns:
            list: list of S&P500 stocks
        """

        control_df = pd.read_csv(
            './data/1_work_data/SP500.csv',
            sep=';'
        )

        active_stocks = control_df.loc[
            control_df['Status'] == True,  # noqa: E712
            'Symbol'
        ].values.tolist()

        return active_stocks

    def extract_sp_500(self):
        """
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

    def extract_stock_data(self, active=True):
        """
        Extract stock all stock data from pre-defined data sources.

        Args:
            active (bool, optional): Set to only extract active
            S&P500 constituents. Defaults to True.
        """

        logger.info("Start stock data extraction process")

        if active:
            stocks = self.active_stocks
        else:
            stocks = self.stocks

        pl_bar = tqdm(total=len(stocks), desc='Stock progress')
        
        for tic in stocks:
            logger.info(f"retrieving {tic} data")

            # create stock instance
            stock = Stock(tic)

            # update stock data
            stock.update_stock_data(method='yh')
            
            # update progress
            pl_bar.update(1)

    def load_stock_data(self):

        logger.info("loading stock data")

        fin_data = []
        mkt_data = []

        for tic in self.stocks[:3]:
            # create stock and retrieve data
            stock = Stock(tic)

            # append stock data
            fin_data.append(stock.get_financial_data())
            mkt_data.append(stock.get_market_data())

        return pd.concat(fin_data), pd.concat(mkt_data)

    def load_index_data(self):

        curr_data_path = list(self.base_data_path.glob('SP500_price.csv'))[0]
        df = pd.read_csv(
            curr_data_path,
            parse_dates=['Date'],
            date_format='%Y-%m-%d'
        )
        return df
