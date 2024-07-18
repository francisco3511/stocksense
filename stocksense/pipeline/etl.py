import os
import json
from pathlib import Path
import datetime as dt

import pandas as pd
from loguru import logger
from tqdm import tqdm

from database_handler import DatabaseHandler
from config import get_config
from utils import (
    scrape_sp500_stock_info,
    get_financial_data,
    get_stock_metadata,
    get_market_data,
    get_stock_insider_data
)


class Etl:
    """
    ETL handler class for stock data.
    Handles data extraction process from each source
    and ingestion process.
    """

    def __init__(self, stocks: list[str | None] = []):
        self.db_handler = DatabaseHandler()
        self.db_fields = get_config("data")["db"]
        self.base_date = get_config("data")["base_date"]
        self.fin_source = "yahoo"
        self.historical_data_path = Path('data/1_work_data/')
        self.stocks = None
        self.set_stocks(stocks)

    def set_stocks(self, stocks: list[str | None] = []) -> list[str]:
        """
        Set stock tickers for ETL process. If none are given,
        sets to all current S&P500 stocks.

        Parameters
        ----------
        stocks : list
            List of tickers to extract data from.

        Returns
        -------
        list[str]
            List of stocks tickers for ETL process.
        """

        logger.info("setting stock data")

        if stocks:
            self.stocks = stocks
        else:
            # read main table and return current index members
            stock_data = self.db_handler.fetch_stock_info()
            self.stocks = stock_data.loc[
                stock_data.active == 1,
                'tic'
            ].values.tolist()

    def update_index_listings(self) -> None:
        """
        Extract updated S&P500 constituents data and update status
        on main stock table.
        """

        logger.info("updating S&P500 index listings")

        # read main stock control table
        hist_df = self.db_handler.fetch_stock_info()

        # extract active sp500 stocks/sectors
        active_df = scrape_sp500_stock_info()

        # get historical, current and last sp500 tickers
        hist_constituents = hist_df['tic'].values.tolist()
        last_constituents = hist_df.loc[
            hist_df.spx_status == 1,
            'tic'
        ].values.tolist()
        constituents = active_df['tic'].values.tolist()

        # downgrade delisted symbols
        for tic in [t for t in last_constituents if t not in constituents]:
            self.db_handler.update_stock_data(tic, {"spx_status": 0})
            logger.info(f"delisted {tic} from S&P500")

        # add new symbols
        for tic in [t for t in constituents if t not in last_constituents]:
            # if there were historical records, reflag as constituent
            if tic in hist_constituents:
                self.db_handler.update_stock_data(
                    tic, {"spx_status": 1, "active": 1}
                )
            else:
                stock_info = active_df.loc[active_df.tic == tic]
                stock_info["last_update"] = dt.datetime.strptime(
                    self.base_date,
                    '%Y-%m-%d'
                ).date()
                stock_info["spx_status"] = 1
                stock_info["active"] = 1
                self.db_handler.insert_stock_info(
                    stock_info[self.db_fields["stocks"]]
                )
            logger.info(f"added {tic} to S&P500 index")

        return None

    def is_empty(self):
        """Check if no stocks were assigned to ETL process."""
        return not self.stocks

    def extract(self):
        """
        Extract stock all stock data from pre-defined data sources.
        """
        logger.info("start stock data extraction process")

        pl_bar = tqdm(total=len(self.stocks), desc='Stock', leave=True)

        if self.is_empty():
            raise Exception("No stocks assigned for ETL process.")

        # update index daily price data
        self.extract_sp_500()

        # update stock data
        for tic in self.stocks:
            self.extract_stock_data(tic)
            pl_bar.update(1)
        pl_bar.close()

    def extract_sp_500(self) -> None:
        """
        Retrieve updated S&P500 data.
        """
        logger.info("extracting S&P500 data")

        try:
            # scrape index data and save to db
            data = get_market_data('^GSPC', self.base_date)
            data = data.drop(columns=["tic"])
            self.db_handler.insert_sp_data(data)
            logger.info("inserted S&P500 market data")
        except Exception:
            logger.error("S&P500 data extraction FAILED")
        return

    def extract_stock_data(self, tic: str) -> bool:
        """
        Extract updated data for a single stock, including
        market, financial and insider trading data.

        Parameters
        ----------
        tic : str
            Ticker of stock to update.
        """
        logger.info(f"extracting {tic} stock data")

        try:
            stock_info = self.db_handler.fetch_stock_info(tic)
            last_update = dt.datetime.strptime(
                stock_info[3], '%Y-%m-%d'
            ).date()
        except Exception:
            # if stock information is not stored on main table, raise exception
            logger.error(f"no stock {tic} info available.")
            return False

        # extract financial data
        if not self.extract_fundamental_data(tic, last_update):
            # if no data found and no data for past 2yrs, flag as innactive
            if last_update.year < dt.datetime.now().year - 2:
                self.db_handler.update_stock_data(tic, {'active': 0})
                logger.info(f"flagged {tic} as innactive")
                return False

        # extract metadata and update stock status
        if not self.extract_metadata(tic):
            # if no metadata found and no data for past yr, flag as innactive
            if last_update.year < dt.datetime.now().year - 1:
                self.db_handler.update_stock_data(tic, {'active': 0})
                logger.info(f"flagged {tic} as innactive")
            return False

        # extract market data
        self.extract_market_data(tic, last_update)
        self.extract_insider_data(tic, last_update)
        return True

    def extract_metadata(self, tic: str) -> bool:
        """
        Extract stock metadata, with relevant current information.

        Parameters
        ----------
        tic : str
            Target stock ticker.

        Returns
        -------
        bool
            Success status.
        """

        try:
            # get stock metadata (adds contextual information about stock)
            info = get_stock_metadata(tic)
            # update market data on db
            self.db_handler.insert_metadata(info)
            return True
        except Exception:
            logger.warning(f"metadata extraction FAILED ({tic})")
            return False

    def extract_fundamental_data(self, tic: str, last_update: dt.date) -> bool:
        """
        Extract financial statement data and update database
        financials table.

        Parameters
        ----------
        tic : str
            Target stock ticker.
        last_update : dt.date
            Date of last data update.

        Returns
        -------
        bool
            Success status.
        """

        try:
            # parse table dates
            fin_data = self.db_handler.fetch_financial_data(tic)
            if fin_data.empty:
                raise Exception("no financial data available")
            start_dt = fin_data['rdq'].max()
        except Exception:
            # no past data available for stock
            fin_data = pd.DataFrame(columns=self.db_fields["financials"])
            logger.warning(
                f'no past financial data found for {tic} ({last_update})'
            )
            start_dt = dt.datetime.strptime(self.base_date, '%Y-%m-%d').date()

        # set end date to present
        end_dt = dt.datetime.now().date()

        # check if earnings season is possible
        if (end_dt - start_dt) < dt.timedelta(days=75):
            logger.warning(
                f'earnings season not reached for {tic} ({last_update})'
            )
            return False
        try:
            # scrape fundamental data
            data = get_financial_data(
                tic,
                start_dt,
                end_dt,
                method=self.fin_source
            )

            # update financial data on db
            self.db_handler.insert_financial_data(
                data[self.db_fields["financials"]]
            )
            self.db_handler.update_stock_data(tic, {'last_update': end_dt})
            logger.info(
                f'updated financial data for {tic} ({start_dt} : {end_dt})'
            )
            return True
        except Exception:
            logger.error(
                f"financial data extraction ({self.fin_source}) FAILED ({tic})"
            )
            return False

    def extract_market_data(self, tic: str, last_update: dt.date) -> bool:
        """
        Extract daily market data, including adjusted close, close
        and volume.

        Parameters
        ----------
        tic : str
            Target stock ticker.
        last_update : dt.date
            Date of last data update.

        Returns
        -------
        bool
            Success status.
        """

        # set end date to present
        end_date = dt.datetime.now().date()

        try:
            if end_date <= last_update:
                logger.info(
                    f'market data already up to date for {tic} ({end_date})'
                )
                return False

            # scrape market data (takes in date as str)
            data = get_market_data(
                tic,
                self.base_date
            )
            # push update to db
            self.db_handler.insert_market_data(
                data[self.db_fields["market"]]
            )
            logger.info(f"updated market data for {tic} ({end_date})")
            return True
        except Exception:
            logger.error(f"market data extraction FAILED ({tic})")
            return False

    def extract_insider_data(self, tic: str, last_update: dt.date) -> bool:
        """
        Extract insider trading data.

        Parameters
        ----------
        tic : str
            Target stock ticker.
        last_update : dt.date
            Date of last data update.

        Returns
        -------
        bool
            Success status (only if new data is inserted).
        """

        # set end date to present
        end_date = dt.datetime.now().strftime('%Y-%m-%d')

        try:
            if dt.datetime.now().date() <= last_update:
                logger.info(
                    f'insider data: already up to date for {tic} ({end_date})'
                )
                return False

            # scrape insider data
            data = get_stock_insider_data(tic)

            # push updated market data to db
            self.db_handler.insert_insider_data(
                data[self.db_fields["insider"]]
            )
            logger.info(f"updated insider trading data for {tic} ({end_date})")
            return True
        except Exception:
            logger.error(f"insider data extraction FAILED ({tic})")
            return False

    def ingest_all_historical_data(self):
        """
        Ingest historical stock data stored in .csv files.
        """

        # read snapshot of S&P500 constituents and store in stocks info table
        self._ingest_stock_info()

        # iterate over stock historical and ingest it
        base_folder = self.historical_data_path / 'company_data'
        for stock_folder in os.listdir(base_folder):
            stock_path = base_folder / stock_folder
            if os.path.isdir(stock_path):
                self._ingest_historical_stock_data(stock_folder, stock_path)

    def _ingest_stock_info(self) -> None:
        """
        Ingest historical S&P500 member info.
        """
        index_df = pd.read_csv(
            self.historical_data_path / 'SP500.csv', sep=';'
        )
        index_df["spx_status"] = index_df["spx_status"].astype('Int64')
        index_df["active"] = index_df["spx_status"]
        index_df["last_update"] = dt.datetime.strptime(
            self.base_date,
            '%Y-%m-%d'
        ).date()
        index_df = index_df[self.db_fields["stocks"]]
        self.db_handler.insert_stock_info(index_df)

    def _ingest_historical_stock_data(
        self,
        tic: str,
        stock_path: Path
    ) -> None:
        """
        Ingest historical stock data, which consists on a snapshot of
        financial, market and insider trading records from historical
        S&P500 members

        Parameters
        ----------
        tic : str
            Target stock ticker.
        stock_path : Path
            Path to target stock data folder.
        """
        try:
            market_file = list(stock_path.glob('market_*.csv'))[0]
            if market_file.exists():
                self._ingest_market_data(market_file, tic)
        except (IndexError, FileNotFoundError) as e:
            logger.warning(f"Market data file not found for {tic}: {e}")

        try:
            insider_file = list(stock_path.glob('insider_*.csv'))[0]
            if insider_file.exists():
                self._ingest_insider_data(insider_file, tic)
        except (IndexError, FileNotFoundError) as e:
            logger.warning(f"Insider data file not found for {tic}: {e}")

        try:
            financials_file = list(stock_path.glob('fundamentals_*.csv'))[0]
            if financials_file.exists():
                # get date of last update and imsert on db
                last_update = dt.datetime.strptime(
                    financials_file.stem.split('_')[1], '%Y-%m-%d'
                ).date()
                self.db_handler.update_stock_data(
                    tic, {'last_update': last_update}
                )
                if last_update.year == dt.datetime.now().date().year:
                    self.db_handler.update_stock_data(tic, {'active': 1})
                self._ingest_financials_data(financials_file, tic)
        except (IndexError, FileNotFoundError) as e:
            logger.warning(f"financials file not found for {tic}: {e}")

        try:
            metadata_file = stock_path / 'metadata.json'
            if metadata_file.exists():
                self._ingest_metadata(metadata_file, tic)
        except (IndexError, FileNotFoundError) as e:
            logger.warning(f"metadata file not found for {tic}: {e}")

    def _ingest_market_data(self, market_file: Path, tic: str) -> None:
        """
        Ingest market data from .csv file.

        Parameters
        ----------
        market_file : Path
            Path to .csv file.
        tic : str
            Target stock ticker.
        """
        try:
            market_df = pd.read_csv(market_file)
            market_df["tic"] = tic
            market_df = market_df.rename(columns={
                "Date": "date",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume"
            })
            market_df = market_df[self.db_fields["market"]]
            market_df['date'] = pd.to_datetime(
                market_df['date'],
                format='ISO8601'
            ).dt.date
            self.db_handler.insert_market_data(market_df)
        except Exception:
            logger.warning(f"market data file for {tic} is empty.")

    def _ingest_insider_data(self, insider_file: Path, tic: str) -> None:
        """
        Ingest insider trading data from .csv file.

        Parameters
        ----------
        insider_file : Path
            Path to .csv file.
        tic : str
            Target stock ticker.
        """
        try:
            insider_df = pd.read_csv(insider_file)
            insider_df['trade_date'] = pd.to_datetime(
                insider_df['trade_date'],
                format='ISO8601'
            ).dt.date
            insider_df['filling_date'] = pd.to_datetime(
                insider_df['filling_date'],
                format='ISO8601'
            ).dt.date
            insider_df["tic"] = tic
            insider_df = insider_df.rename(columns={
                "Title": "title",
                "Qty": "qty",
                "Owned": "owned",
                "Value": "value",
            })
            insider_df = insider_df[self.db_fields["insider"]]
            self.db_handler.insert_insider_data(insider_df)
        except Exception:
            logger.warning(f"insider data file for {tic} is empty.")

    def _ingest_financials_data(self, financials_file: Path, tic: str) -> None:
        """
        Ingest financial data from .csv file.

        Parameters
        ----------
        financials_file : Path
            Path to .csv file.
        tic : str
            Target stock ticker.
        """
        try:
            financials_df = pd.read_csv(financials_file)
            financials_df = financials_df[self.db_fields["financials"]]
            financials_df['datadate'] = pd.to_datetime(
                financials_df['datadate'],
                format='ISO8601'
            ).dt.date
            financials_df['rdq'] = pd.to_datetime(
                financials_df['rdq'],
                format='ISO8601'
            ).dt.date
            self.db_handler.insert_financial_data(financials_df)
        except Exception:
            logger.warning(f"financials data file for {tic} is empty.")

    def _ingest_metadata(self, metadata_file: Path, tic: str) -> None:
        """
        Ingest stock metadata from JSON file.

        Parameters
        ----------
        metadata_file : Path
            Path to stock metadata JSON file.
        tic : str
            Target stock ticker.
        """
        try:
            record = {
                'tic': tic,
                'shares_outstanding': None,
                'enterprise_value': None,
                'rec_key': None,
                'forward_pe': None
            }
            # read metadata file and try to load contents
            f = open(metadata_file)
            data = json.load(f)
            if 'sharesOutstanding' in data:
                record['shares_outstanding'] = data['sharesOutstanding']
            if 'enterpriseValue' in data:
                record['enterprise_value'] = data['enterpriseValue']
            if 'recommendationKey' in data:
                record['rec_key'] = data['recommendationKey']
            if 'forwardPE' in data:
                record['forward_pe'] = data['forwardPE']
            self.db_handler.insert_metadata(record)
        except Exception:
            logger.warning(f"metadata file for {tic} is empty.")
