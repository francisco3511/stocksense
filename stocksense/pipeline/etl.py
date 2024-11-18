import datetime as dt
import os
from pathlib import Path
from typing import Optional

import polars as pl
from loguru import logger
from tqdm import tqdm

from stocksense.database_handler import DatabaseHandler

from .scraper import Scraper

PACKAGE_DIR = Path(__file__).parents[1]
DATA_PATH = PACKAGE_DIR / "data"


class ETL:
    """
    ETL handler class for stock data.
    Handles remote source data extraction,
    transformation and DB ingestion processes.
    """

    def __init__(self, config, stocks: Optional[list[str]] = None):
        self.db: DatabaseHandler = DatabaseHandler()
        self.db_schema: dict = config.database.db_schema
        self.base_date: str = config.scraping.base_date
        self.fin_source: str = "yfinance"
        self.historical_data_path: Path = DATA_PATH / "interim"
        self.stocks: list[str] = stocks or self._set_default_stocks()

    def _set_default_stocks(self) -> list[str]:
        """
        Retrieve default S&P500 stock tickers if none are provided.

        Returns
        -------
        list[str]
            List of stocks tickers for ETL process.
        """
        logger.info("setting default S&P500 stock tickers")
        stock_data = self.db.fetch_stock()
        if stock_data.is_empty():
            self.set_index_listings()
        return stock_data.filter(pl.col("spx_status") == 1)["tic"].to_list()

    def set_index_listings(self) -> None:
        """
        Set index stock control table.
        """
        df = Scraper.scrape_sp500_stock_info()
        parsed_date = dt.datetime.strptime(self.base_date, "%Y-%m-%d").date()
        stock_df = df.with_columns(
            [
                pl.lit(parsed_date).alias("last_update"),
                pl.lit(1).alias("spx_status"),
                pl.lit(1).alias("active"),
            ]
        )
        self.db.insert_stock(stock_df[self.db_schema["stock"]])

    def update_index_listings(self) -> None:
        """
        Update the S&P500 index constituents in the database.
        """
        logger.info("updating S&P500 index listings")

        stock_df = self.db.fetch_stock()
        active_df = Scraper.scrape_sp500_stock_info()

        self._update_delisted_symbols(stock_df, active_df)
        self._add_new_symbols(stock_df, active_df)

    def _update_delisted_symbols(self, stock_df: pl.DataFrame, active_df: pl.DataFrame) -> None:
        """
        Downgrade delisted symbols from S&P500.

        Parameters
        ----------
        stock_df : pl.DataFrame
            stock control table.
        active_df : pl.DataFrame
            current sp500 stocks and sectors.
        """
        last_constituents = stock_df.filter(pl.col("spx_status") == 1)["tic"].to_list()
        constituents = active_df["tic"].to_list()

        for tic in [t for t in last_constituents if t not in constituents]:
            self.db.update_stock(tic, {"spx_status": 0})
            logger.info(f"{tic}: delisted from S&P500")

    def _add_new_symbols(self, stock_df: pl.DataFrame, active_df: pl.DataFrame) -> None:
        """
        Adds new symbols to S&P500.

        Parameters
        ----------
        stock_df : pl.DataFrame
            Stock control table.
        active_df : pl.DataFrame
            Current sp500 stocks and sectors.
        """
        hist_constituents = stock_df["tic"].to_list()
        last_constituents = stock_df.filter(pl.col("spx_status") == 1)["tic"].to_list()
        constituents = active_df["tic"].to_list()

        for tic in [t for t in constituents if t not in last_constituents]:
            if tic in hist_constituents:
                self.db.update_stock(tic, {"spx_status": 1, "active": 1})
            else:
                parsed_date = dt.datetime.strptime(self.base_date, "%Y-%m-%d").date()
                stock = active_df.filter(pl.col("tic") == tic).with_columns(
                    [
                        pl.lit(parsed_date).alias("last_update"),
                        pl.lit(1).alias("spx_status"),
                        pl.lit(1).alias("active"),
                    ]
                )
                self.db.insert_stock(stock[self.db_schema["stock"]])

            logger.info(f"added {tic} to S&P500 index")

    def is_empty(self):
        """
        Check if no stocks were assigned to ETL process.
        """
        return not self.stocks

    def extract(self) -> None:
        """
        Extract all stock data from pre-defined data sources.
        """
        logger.info("START stock data extraction process")
        if self.is_empty():
            raise ValueError("No stocks assigned for ETL process.")
        self.extract_sp_500()
        self.extract_vix()
        self.extract_all_stocks()

    def extract_sp_500(self) -> None:
        """
        Retrieve updated S&P500 data.
        """
        logger.info("extracting S&P500 data")
        try:
            scraper = Scraper("^GSPC", self.fin_source)
            data = scraper.get_market_data(self.base_date)
            self.db.insert_index_data(data[self.db_schema["sp500"]])
            logger.info("inserted S&P500 market data")
        except Exception:
            logger.error("S&P500 data extraction FAILED")
        return

    def extract_vix(self) -> None:
        """
        Retrieve updated daily VIX data.
        """
        logger.info("extracting VIX data")
        try:
            scraper = Scraper("^VIX", self.fin_source)
            data = scraper.get_market_data(self.base_date)
            self.db.insert_vix_data(data[self.db_schema["vix"]])
            logger.info("inserted VIX market data")
        except Exception:
            logger.error("VIX data extraction FAILED")
        return

    def extract_all_stocks(self) -> None:
        """
        Extract data for all assigned stocks.
        """
        pl_bar = tqdm(total=len(self.stocks), desc="Stock", leave=True)
        for tic in self.stocks:
            self.extract_stock_data(tic)
            pl_bar.update(1)
        pl_bar.close()

    def extract_stock_data(self, tic: str) -> bool:
        """
        Extract updated data for a single stock, including market, financial and
        insider trading data. If no financial data is found for the last 2 years,
        flags stock as delisted.


        Parameters
        ----------
        tic : str
            Ticker of stock to update.

        Returns
        -------
        bool
            Sucess status
        """
        logger.info(f"START {tic} stock data extraction")

        try:
            stock = self.db.fetch_stock(tic).row(0, named=True)
            last_update = stock["last_update"]
        except Exception:
            logger.error(f"{tic}: no stock {tic} info available.")
            return False

        scraper = Scraper(tic, self.fin_source)
        if not self.extract_fundamental_data(tic, scraper, last_update):
            if last_update.year < dt.datetime.now().year - 2:
                self.db.update_stock(tic, {"active": 0})
                logger.info(f"{tic}: flagged as inactive")
                return False

        self.extract_info(tic, scraper)
        self.extract_market_data(tic, scraper)
        self.extract_insider_data(tic, scraper)
        return True

    def extract_info(self, tic: str, scraper: Scraper) -> bool:
        """
        Extract stock info, with relevant current information.

        Parameters
        ----------
        tic : str
            Target stock ticker.
        scraper : Scraper
            Stock data scraping handler object.

        Returns
        -------
        bool
            Success status.
        """
        try:
            info = scraper.get_stock_info()
            self.db.insert_info(info)
            self.db.update_stock(tic, {"last_update": dt.datetime.now().date()})
            logger.info(f"{tic}: updated stock info")
            return True
        except Exception:
            logger.error(f"{tic}: info extraction FAILED")
            return False

    def extract_fundamental_data(self, tic: str, scraper: Scraper, last_update: dt.date) -> bool:
        """
        Extract financial statement data and update database
        financials table.

        Parameters
        ----------
        tic : str
            Target stock ticker.
        scraper : Scraper
            Stock data scraping handler object.
        last_update : dt.date
            Date of last data update.

        Returns
        -------
        bool
            Success status.
        """

        try:
            fin_data = self.db.fetch_financial_data(tic)
            if fin_data.is_empty():
                raise Exception(f"{scraper.tic}: no financial data available")
            start_date = fin_data["datadate"].max()
        except Exception:
            # no past data available for stock
            fin_data = pl.DataFrame(schema=self.db_schema["financial"])
            start_date = dt.datetime.strptime(self.base_date, "%Y-%m-%d").date()
            logger.warning(f"{tic}: no past financial data found ({last_update})")

        try:
            end_date = dt.datetime.now().date()
            if (end_date - start_date) < dt.timedelta(days=80):
                logger.warning(f"{tic}: earnings season not reached ({last_update})")
                return False

            data = scraper.get_financial_data(start_date, end_date)
            self.db.insert_financial_data(data[self.db_schema["financial"]])
            self.db.update_stock(tic, {"last_update": end_date})
            logger.success(f"{tic}: updated financial data ({start_date}:{end_date})")
            return True
        except Exception as e:
            logger.error(f"{tic}: financial data extraction FAILED ({e})")
            return False

    def extract_market_data(self, tic: str, scraper: Scraper) -> bool:
        """
        Extract daily market data, including adjusted close, close
        and volume.

        Parameters
        ----------
        tic : str
            Target stock ticker.
        scraper : Scraper
            Stock data scraping handler object.
        last_update : dt.date
            Date of last data update.

        Returns
        -------
        bool
            Success status.
        """
        try:
            end_date = dt.datetime.now().date()
            mkt_data = self.db.fetch_market_data(tic)

            if not mkt_data.is_empty():
                if end_date <= mkt_data["date"].max():
                    logger.info(f"{tic}: market data already up to date ({end_date})")
                    return False

            data = scraper.get_market_data(self.base_date)
            self.db.insert_market_data(data[self.db_schema["market"]])
            self.db.update_stock(tic, {"last_update": end_date})
            logger.success(f"{tic}: updated market data ({end_date})")
            return True
        except Exception as e:
            logger.error(f"{tic}: market data extraction FAILED ({e})")
            return False

    def extract_insider_data(self, tic: str, scraper: Scraper) -> bool:
        """
        Extract insider trading data.

        Parameters
        ----------
        tic : str
            Target stock ticker.
        scraper : Scraper
            Stock data scraping handler object.
        last_update : dt.date
            Date of last data update.

        Returns
        -------
        bool
            Success status.
        """
        try:
            end_date = dt.datetime.now().date()
            ins_data = self.db.fetch_insider_data(tic)

            if not ins_data.is_empty():
                if end_date <= ins_data["filling_date"].max():
                    logger.info(f"{tic}: insider data already updated ({end_date})")
                    return False

            data = scraper.get_stock_insider_data()
            self.db.insert_insider_data(data[self.db_schema["insider"]])
            logger.success(f"{tic}: updated insider trading data ({end_date})")
            return True
        except Exception as e:
            logger.error(f"{tic}: insider data extraction FAILED ({e})")
            return False

    def ingest_all_historical_data(self):
        """
        Ingest historical stock data stored in .csv files.
        """

        # read snapshot of S&P500 constituents and store in stocks info table
        self._ingest_stock_list()

        # iterate over stock historical and ingest it
        base_folder = self.historical_data_path / "company_data"
        for stock_folder in os.listdir(base_folder):
            stock_path = base_folder / stock_folder
            if os.path.isdir(stock_path):
                self._ingest_historical_stock_data(stock_folder, stock_path)

    def _ingest_stock_list(self) -> None:
        """
        Ingest historical S&P500 member info.
        """

        index_df = pl.read_csv(self.historical_data_path / "SP500.csv", separator=";")
        parsed_date = dt.datetime.strptime(self.base_date, "%Y-%m-%d").date()

        index_df = index_df.with_columns(
            pl.col("spx_status").cast(pl.Int16),
            pl.col("spx_status").cast(pl.Int16).alias("active"),
            pl.lit(parsed_date).alias("last_update"),
        )[self.db_schema["stock"]]

        self.db.insert_stock(index_df)

    def _ingest_historical_stock_data(self, tic: str, stock_path: Path) -> None:
        """
        Ingest historical stock data, which consists on a snapshot of
        financial, market and insider trading records from historical
        S&P500 members.
        """
        try:
            market_file = list(stock_path.glob("market_*.csv"))[0]
            if market_file.exists():
                self._ingest_market_data(market_file, tic)
        except (IndexError, FileNotFoundError) as e:
            logger.warning(f"market data file not found for {tic}: {e}")

        try:
            insider_file = list(stock_path.glob("insider_*.csv"))[0]
            if insider_file.exists():
                self._ingest_insider_data(insider_file, tic)
        except (IndexError, FileNotFoundError) as e:
            logger.warning(f"insider data file not found for {tic}: {e}")

        try:
            financials_file = list(stock_path.glob("fundamentals_*.csv"))[0]
            if financials_file.exists():
                # get date of last update and imsert on db
                last_update = dt.datetime.strptime(
                    financials_file.stem.split("_")[1], "%Y-%m-%d"
                ).date()
                self.db.update_stock(tic, {"last_update": last_update})
                if last_update.year == dt.datetime.now().date().year:
                    self.db.update_stock(tic, {"active": 1})
                self._ingest_financials_data(financials_file, tic)
        except (IndexError, FileNotFoundError) as e:
            logger.warning(f"financials file not found for {tic}: {e}")

    def _ingest_market_data(self, market_file: Path, tic: str) -> None:
        """
        Ingest market data from .csv file.
        """
        try:
            market_df = pl.read_csv(market_file)
            market_df = market_df.with_columns(
                pl.col("Date").str.to_date("%Y-%m-%d"), pl.lit(tic).alias("tic")
            )
            market_df = market_df.rename(
                {
                    "Date": "date",
                    "Close": "close",
                    "Adj Close": "adj_close",
                    "Volume": "volume",
                }
            )
            market_df = market_df[self.db_schema["market"]]
            self.db.insert_market_data(market_df)
        except Exception:
            logger.warning(f"market data file for {tic} is empty.")

    def _ingest_insider_data(self, insider_file: Path, tic: str) -> None:
        """
        Ingest insider trading data from .csv file.
        """
        try:
            insider_df = pl.read_csv(insider_file)
            insider_df = insider_df.with_columns(
                pl.col("filling_date").str.to_datetime().dt.date(),
                pl.col("trade_date").str.to_datetime().dt.date(),
                pl.lit(tic).alias("tic"),
            )

            insider_df = insider_df.rename(
                {
                    "Title": "title",
                    "Qty": "qty",
                    "Owned": "owned",
                    "Value": "value",
                }
            )
            insider_df = insider_df[self.db_schema["insider"]]
            self.db.insert_insider_data(insider_df)
        except Exception:
            logger.warning(f"insider data file for {tic} is empty.")

    def _ingest_financials_data(self, financials_file: Path, tic: str) -> None:
        """
        Ingest financial data from .csv file.
        """
        try:
            financials_df = pl.read_csv(financials_file)
            financials_df = financials_df.with_columns(
                pl.col("datadate").str.to_date("%Y-%m-%d"),
                pl.col("rdq").str.to_date("%Y-%m-%d"),
                pl.lit(tic).alias("tic"),
            )
            financials_df = financials_df[self.db_schema["financial"]]
            self.db.insert_financial_data(financials_df)
        except Exception:
            logger.warning(f"financials data file for {tic} is empty.")
