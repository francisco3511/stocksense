import datetime as dt
import os
from pathlib import Path
from typing import Optional

import polars as pl
from loguru import logger
from tqdm import tqdm

from stocksense.config import DATA_DIR, ConfigManager
from stocksense.database import DatabaseHandler

from .scraper import Scraper


class ETL:
    """
    ETL handler class for stock data.
    Handles remote source data extraction,
    transformation and DB ingestion processes.
    """

    def __init__(self, config: ConfigManager, stocks: Optional[list[str]] = None):
        self.db: DatabaseHandler = DatabaseHandler()
        self.db_schema: dict = config.database.db_schema
        self.base_date: str = config.scraping.base_date
        self.fin_source: str = "yfinance"
        self.historical_data_path: Path = DATA_DIR / "interim"
        self._update_index_listings()
        self.stocks: list[str] = stocks or self._set_default_stocks()

    def _update_index_listings(self) -> None:
        """
        Update the S&P500 index constituents in the database.
        """
        logger.info("updating S&P500 control table")

        stock_df = self.db.fetch_stock()
        sp500_df = Scraper.scrape_sp500_constituents()
        additions, removals = Scraper.scrape_sp500_changes()

        last_constituents = stock_df.filter(pl.col("date_removed").is_null())["tic"].to_list()
        current_constituents = sp500_df["tic"].to_list()

        self._delist_stocks(last_constituents, current_constituents, removals)
        self._add_new_stocks(last_constituents, sp500_df)

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
            self._set_index_listings()

        # fetch S&P500 stocks and recently delisted stocks
        stock_data = stock_data.filter(
            pl.col("date_removed").is_null()
            | (pl.col("date_removed") >= (dt.datetime.now().date() - dt.timedelta(days=360)))
        )
        return sorted(stock_data["tic"].to_list())

    def _set_index_listings(self) -> None:
        """
        Set index stock control table.
        """
        sp500_df = Scraper.scrape_sp500_constituents()
        sp500_df = sp500_df.with_columns(pl.lit(None).alias("date_removed"))
        self.db.insert_stock(sp500_df[self.db_schema["stock"]])

    def _delist_stocks(
        self, last_constituents: list[str], current_constituents: list[str], removals: pl.DataFrame
    ) -> None:
        """
        Flag stock as delisted.

        Parameters
        ----------
        last_constituents : list[str]
            Last S&P500 constituents.
        current_constituents : list[str]
            Current S&P500 constituents.
        removals : pl.DataFrame
            S&P500 changes data.
        """
        removals_list = removals["tic"].to_list()
        removed_stocks = []
        for tic in last_constituents:
            if tic not in current_constituents:
                if tic in removals_list:
                    removed_date = dt.datetime.strptime(
                        removals.filter(pl.col("name_removed") == tic)["removed"].to_list()[0],
                        "%Y-%m-%d",
                    ).date()
                    self.db.update_stock(tic, {"date_removed": removed_date})
                else:
                    today = dt.datetime.now().date()
                    self.db.update_stock(tic, {"date_removed": today})
                removed_stocks.append(tic)
                logger.info(f"{tic}: delisted from S&P500")
        logger.info(f"removed {removed_stocks} from S&P500 index")
        return

    def _add_new_stocks(self, last_constituents: list[str], sp500_df: pl.DataFrame) -> None:
        new_stocks = sp500_df.filter(~pl.col("tic").is_in(last_constituents))
        new_stocks = new_stocks.with_columns(pl.lit(None).alias("date_removed"))
        self.db.insert_stock(new_stocks[self.db_schema["stock"]])
        logger.info(f"added {new_stocks['tic'].to_list()} to S&P500 index")
        return

    def is_empty(self) -> bool:
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
        return

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

        if tic not in self.db.fetch_stock()["tic"].to_list():
            logger.info(f"{tic}: not tracked.")
            return False

        scraper = Scraper(tic, self.fin_source)
        self.extract_fundamental_data(tic, scraper)
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
            logger.success(f"{tic}: updated stock info")
            return True
        except Exception:
            logger.error(f"{tic}: info extraction FAILED")
            return False

    def extract_fundamental_data(self, tic: str, scraper: Scraper) -> bool:
        """
        Extract financial statement data and update database
        financials table.

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
            fin_data = self.db.fetch_financial_data(tic)
            if fin_data.is_empty():
                raise Exception(f"{scraper.tic}: no financial data available")
            start_date = fin_data["datadate"].max()
        except Exception:
            # no past data available for stock
            fin_data = pl.DataFrame(schema=self.db_schema["financial"])
            start_date = dt.datetime.strptime(self.base_date, "%Y-%m-%d").date()
            logger.warning(f"{tic}: no past financial data found")

        try:
            end_date = dt.datetime.now().date()
            if (end_date - start_date) < dt.timedelta(days=80):
                logger.warning(f"{tic}: earnings season not reached ({start_date}:{end_date})")
                return False

            data = scraper.get_financial_data(start_date, end_date)
            self.db.insert_financial_data(data[self.db_schema["financial"]])
            logger.success(f"{tic}: updated financial data ({start_date}:{end_date})")
            return True
        except Exception as e:
            logger.error(f"{tic}: financial data extraction FAILED - {e}")
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

    def restore_delisted_stocks_data(self) -> None:
        """
        Restore market data for delisted stocks from historical CSV files.
        This prevents newer data from companies reusing tickers from
        overwriting historical records.
        """

        # Get all delisted stocks
        cutoff_date = dt.date(2018, 1, 1)
        delisted_stocks = (
            self.db.fetch_stock()
            .filter(
                (pl.col("date_removed").is_not_null())
                & (pl.col("date_removed") < cutoff_date)
            )
            ["tic"]
            .to_list()
        )

        logger.info(f"Restoring market data for {delisted_stocks}")
        prices_file = DATA_DIR / "raw" / "prices_2005-01-01_2018-12-31.csv"

        with open(prices_file) as f:
            data_types = f.readline().strip().split(',')[1:]
            tickers = f.readline().strip().split(',')[1:]

        columns = (
            ['date'] +
            [f"{dtype}_{tic}" for dtype, tic in zip(data_types, tickers, strict=True)]
        )

        df = pl.read_csv(
            prices_file,
            has_header=False,
            skip_rows=2,
            new_columns=columns
        )

        df = df.filter(pl.col("date") != "Date")
        stock_cols = ['date']
        for stock in delisted_stocks:
            stock_cols.extend([
                col for col in df.columns
                if col.endswith(f"_{stock}")
                and any(prefix in col for prefix in ["Close", "Adj Close", "Volume"])
            ])

        df = df.select(stock_cols)

        for tic in delisted_stocks:
            try:
                stock_df = df.select([
                    pl.col("date").str.to_date(),
                    pl.col(f"Close_{tic}").cast(pl.Float64).alias("close"),
                    pl.col(f"Adj Close_{tic}").cast(pl.Float64).alias("adj_close"),
                    pl.col(f"Volume_{tic}").cast(pl.Int64).alias("volume"),
                    pl.lit(tic).alias("tic")
                ])
                stock_df = stock_df.filter(pl.col("close").is_not_null())

                if not stock_df.is_empty():
                    self.db.delete_market_data(tic)
                    stock_df = stock_df[["tic", "date", "close", "adj_close", "volume"]]
                    self.db.insert_market_data(stock_df)

            except Exception as e:
                logger.error(f"error restoring market data for {tic}: {e}")
                continue

    def ingest_all_historical_data(self):
        """
        Ingest historical stock data stored in .csv files.
        """
        self._ingest_stock_list()
        base_folder = self.historical_data_path / "company_data"
        for stock_folder in os.listdir(base_folder):
            stock_path = base_folder / stock_folder
            if os.path.isdir(stock_path):
                self._ingest_historical_stock_data(stock_folder, stock_path)

    def _ingest_stock_list(self) -> None:
        """
        Ingest historical S&P500 member info.
        """
        base_df = pl.read_csv(
            self.historical_data_path / "SP500.csv",
            separator=";",
            columns=["tic", "name", "sector"],
        )

        base_df = pl.read_csv(self.historical_data_path / "SP500.csv", separator=";")
        curr_df = Scraper.scrape_sp500_constituents()
        additions, removals = Scraper.scrape_sp500_changes()

        df = base_df.join(additions, on="tic", how="left")
        df = df.join(removals, on="tic", how="left")
        df = df.join(curr_df.select(["tic", "date_added"]), on="tic", how="left")
        df = df.with_columns(pl.col("removed").alias("date_removed"))
        df = df.with_columns(
            pl.when(pl.col("date_added").is_null())
            .then(pl.col("added"))
            .otherwise(pl.col("date_added"))
            .alias("date_added")
        ).with_columns(
            pl.when(pl.col("name").is_null())
            .then(pl.col("name_removed"))
            .otherwise(pl.col("name"))
            .alias("name")
        )
        df = df.filter(~(pl.col("date_added").is_null() & pl.col("date_removed").is_null()))
        df = df[self.db_schema["stock"]]
        self.db.insert_stock(df)

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
