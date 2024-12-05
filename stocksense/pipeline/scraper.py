import datetime as dt
import logging
import re
import time

import polars as pl
import requests
import yfinance as yf
from bs4 import BeautifulSoup as bs
from pyrate_limiter import Duration, Limiter, RequestRate
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket

from stocksense.config import config

logging.getLogger("yfinance").setLevel(logging.CRITICAL)


class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    pass


class Scraper:
    """
    Stock web scraping class (inc. financial, market, info and insider info)
    Currently supports yfinance source only.
    """

    def __init__(self, tic, source):
        self.tic: str = tic
        self.source: str = source
        self.session: CachedLimiterSession = self._get_session()
        if self.source == "yfinance":
            self.handler: yf.Ticker = self._get_yfinance_handler()

    def _get_session(self):
        """
        Create session for yfinance queries.

        Returns
        -------
        CachedLimiterSession
            Session object.
        """
        session = CachedLimiterSession(
            limiter=Limiter(RequestRate(2, Duration.SECOND * 5)),
            bucket_class=MemoryQueueBucket,
            backend=SQLiteCache(f"{self.source}.cache"),
        )
        session.headers["User-agent"] = "my-program/1.0"
        return session

    def _get_yfinance_handler(self):
        """
        Get yfinance stock ticker handler.
        """
        return yf.Ticker(self.tic, session=self.session)

    def _get_market_data_yfinance(self, start_date):
        """
        Scrape daily market data for a stock, until present.
        """
        df = pl.from_pandas(
            self.handler.history(start=start_date, auto_adjust=False).reset_index(drop=False)
        )

        if df.is_empty():
            raise Exception("No market data available.")

        df = df.with_columns(
            [
                pl.col("Date").dt.date().alias("date"),
                pl.col("Close").alias("close"),
                pl.col("Adj Close").alias("adj_close"),
                pl.col("Volume").alias("volume"),
                pl.lit(self.tic).alias("tic"),
            ]
        )
        df = df.select(["date", "close", "adj_close", "volume", "tic"])
        return df

    def _get_stock_info_yfinance(self) -> dict:
        """
        Scrape current info using yfinance.

        Returns
        -------
        dict
            Current stock info.

        Raises
        ------
        Exception
            No info available.
        """
        data = self.handler.info

        if not data:
            raise Exception("No status information data available.")

        fields = config.scraping.yahoo_info
        record = dict.fromkeys(list(fields.values()), None)
        record["tic"] = self.tic

        for yh_key, key in fields.items():
            if yh_key in data:
                record[key] = data[yh_key]

        return record

    def _get_fundamental_data_yfinance(
        self, start_date: dt.date, end_date: dt.date
    ) -> pl.DataFrame:
        """
        Scrape fundamental data from Yahoo Finance using yfinance lib, searching
        for financial records released between two dates.

        Parameters
        ----------
        start_date : dt.date
            Starting date.
        end_date : dt.date
            Ending date.

        Returns
        -------
        pl.DataFrame
            Financial report data from yfinance.

        Raises
        ------
        Exception
            No financial data available for date interval.
        """

        fields_to_keep = config.scraping.yahoo

        # retrieve 3 main financial documents
        is_df = pl.from_pandas(self.handler.quarterly_income_stmt.T.reset_index())
        bs_df = pl.from_pandas(self.handler.quarterly_balance_sheet.T.reset_index())
        cf_df = pl.from_pandas(self.handler.quarterly_cashflow.T.reset_index())

        # parse dates
        is_df = is_df.with_columns(pl.col("index").dt.date().alias("index")).sort("index")
        bs_df = bs_df.with_columns(pl.col("index").dt.date().alias("index")).sort("index")
        cf_df = cf_df.with_columns(pl.col("index").dt.date().alias("index")).sort("index")

        df = is_df.join_asof(
            bs_df, on="index", strategy="backward", tolerance=dt.timedelta(days=30)
        ).join_asof(cf_df, on="index", strategy="backward", tolerance=dt.timedelta(days=30))

        for c in list(fields_to_keep.keys()):
            if c not in df.columns:
                df = df.with_columns(pl.lit(None).alias(c))

        df = df.select(list(fields_to_keep.keys()))
        df = df.rename(fields_to_keep)
        df = df.filter((pl.col("datadate") > start_date) & (pl.col("datadate") <= end_date))

        if df.is_empty():
            raise Exception("No financial data available for date interval.")

        for c in df.columns[1:]:
            df = df.with_columns(pl.col(c).cast(pl.Float64))
            df = df.with_columns((pl.col(c) / 1000000).round(3).alias(c))

        df = df.with_columns([(-pl.col("dvq")).alias("dvq"), (-pl.col("capxq")).alias("capxq")])
        df = df.unique(subset=["datadate"]).sort("datadate")
        df = df.with_columns(pl.lit(self.tic).alias("tic"))
        return df

    def _get_earnings_dates_yfinance(self, start_date: dt.date, end_date: dt.date):
        """
        Scrape earnings dates and eps surprise.
        """
        try:
            n_quarters = int((end_date - start_date).days / 90) + 20
            df = pl.from_pandas(self.handler.get_earnings_dates(limit=n_quarters).reset_index())
            df = df.rename({"Earnings Date": "rdq", "Surprise(%)": "surprise_pct"})
            df = df.select(["rdq", "surprise_pct"])
            df = df.with_columns(pl.col("rdq").dt.date())
            df = df.filter((pl.col("rdq") >= start_date) & (pl.col("rdq") <= end_date))
            df = df.unique(subset=["rdq"]).sort("rdq").drop_nulls(subset=["surprise_pct", "rdq"])
            if df.is_empty():
                raise pl.exceptions.EmptyDataFrame("No financial release date available.")
        except Exception:
            return pl.DataFrame()

    def _get_earnings_dates_sec(self, start_date: dt.date, end_date: dt.date):
        """
        Scrape earnings dates from SEC.
        """
        base_url = "https://www.sec.gov/cgi-bin/browse-edgar"
        headers = {
            "User-Agent": "Company Name AdminContact@domain.com",  # Replace with your details
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov",
        }
        params = {
            "action": "getcompany",
            "CIK": self.tic,
            "type": "10-",
            "owner": "include",
            "count": "100",
        }
        try:
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()
            soup = bs(response.text, "html.parser")

            time.sleep(1)

            if soup.select("p > center > h1"):
                return pl.DataFrame()

            dates = []
            dateFind = re.compile(r"2\d{3}-\d{2}-\d{2}")

            for tr in soup.select("tr"):
                tdElems = tr.select("td")
                if len(tdElems) == 5 and dateFind.search(tdElems[3].getText()):
                    date = tdElems[3].getText().strip()
                    dates.append(date)

            if not dates:
                return pl.DataFrame()

            df = pl.DataFrame({"rdq": dates, "surprise_pct": [None] * len(dates)})
            df = df.with_columns(pl.col("rdq").str.to_date("%Y-%m-%d"))
            df = df.filter((pl.col("rdq") >= start_date) & (pl.col("rdq") <= end_date))
            return df.unique(subset=["rdq"]).sort("rdq")

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return pl.DataFrame()

    def get_market_data(self, start_date):
        """
        Scrape market data.
        """
        if self.source == "yfinance":
            return self._get_market_data_yfinance(start_date)
        else:
            raise Exception("Other methods not implemented")

    def get_stock_info(self):
        """
        Scrape current stock info.
        """
        if self.source == "yfinance":
            return self._get_stock_info_yfinance()
        else:
            raise Exception("Other methods not implemented")

    def get_financial_data(
        self,
        start_date: dt.date,
        end_date: dt.date,
    ) -> pl.DataFrame:
        """
        Scrape financial data.
        """
        if self.source == "yfinance":
            df = self._get_fundamental_data_yfinance(start_date, end_date)
            earn_dates = self._get_earnings_dates_sec(start_date, end_date)
        else:
            raise Exception("Other methods not implemented")

        try:
            df = df.with_columns(pl.col("datadate").dt.date()).sort("datadate")
            earn_dates = earn_dates.with_columns(pl.col("rdq").dt.date()).sort("rdq")
            df = df.join_asof(
                earn_dates,
                left_on="datadate",
                right_on="rdq",
                strategy="forward",
                tolerance=dt.timedelta(days=80),
            )
        except pl.exceptions.ColumnNotFoundError:
            df = df.with_columns([pl.lit(None).alias("rdq"), pl.lit(None).alias("surprise_pct")])

        df = df.with_columns(
            pl.when(pl.col("rdq").is_null())
            .then(pl.col("datadate") + pl.duration(days=80))
            .otherwise(pl.col("rdq"))
            .alias("rdq")
        )
        df = df.with_columns(
            pl.when(pl.col("rdq") > dt.datetime.now().date())
            .then(dt.datetime.now().date())
            .otherwise(pl.col("rdq"))
            .alias("rdq")
        )
        return df

    def get_stock_insider_data(self) -> pl.DataFrame:
        """
        Scrape OpenInsider.com insider trading data for a given stock.
        """

        field_names = [
            "filling_date",
            "trade_date",
            "owner_name",
            "title",
            "transaction_type",
            "last_price",
            "qty",
            "shares_held",
            "owned",
            "value",
        ]
        # set url format for OpenInsider.com
        url = (
            f"http://openinsider.com/screener?s={self.tic}&o=&pl=&ph=&ll=&lh=&fd=0"
            "&fdr=&td=0&tdr=&fdlyl=&fdlyh=&daysago=&xp=1&xs=1&vl=&vh=&ocl=&och=&"
            "sic1=-1&sicl=100&sich=9999&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&"
            "v2h=&oc2l=&oc2h=&sortcol=0&cnt=1000&page=1"
        )
        try:
            data = []
            response = requests.get(url)
            soup = bs(response.text, "html.parser")
            table = soup.find("table", {"class": "tinytable"})
            if not table:
                raise ValueError("Table not found on page")

            rows = table.find_all("tr")
            for row in rows[1:]:
                cols = row.find_all("td")
                if not cols:
                    continue
                insider_data = [
                    cols[1].text.strip(),
                    cols[2].text.strip(),
                    cols[4].text.strip(),
                    cols[5].text.strip(),
                    cols[6].text.strip(),
                    cols[7].text.strip(),
                    cols[8].text.strip(),
                    cols[9].text.strip(),
                    cols[10].text.strip(),
                    cols[11].text.strip(),
                ]
                data.append(insider_data)

            df = pl.DataFrame(data, schema=field_names, orient="row")
            df = df.with_columns(pl.lit(self.tic).alias("tic"))

            if df.is_empty():
                raise Exception("No insider data available")

            df = df.with_columns(
                [
                    pl.col("filling_date").str.to_datetime("%Y-%m-%d %H:%M:%S").dt.date(),
                    pl.col("trade_date").str.to_date("%Y-%m-%d"),
                ]
            )
            df = df.sort("filling_date")
            return df
        except Exception as e:
            raise e

    @staticmethod
    def scrape_sp500_constituents() -> pl.DataFrame:
        """
        List S&P500 stock info from wiki page and return Polars dataframe.
        """
        resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        soup = bs(resp.text, "lxml")
        table = soup.find("table", id="constituents")

        data = {"tic": [], "name": [], "sector": [], "date_added": []}

        for row in table.findAll("tr")[1:]:
            ticker = row.findAll("td")[0].text.replace("\n", "")
            security = row.findAll("td")[1].text.replace("\n", "")
            sector = row.findAll("td")[2].text.replace("\n", "")
            date_add = row.findAll("td")[5].text.replace("\n", "")
            data["tic"].append(ticker)
            data["name"].append(security)
            data["sector"].append(sector)
            data["date_added"].append(date_add)

        df = pl.DataFrame(data)
        df = df.with_columns(pl.col("tic").str.replace(".", "-", literal=True))
        df = df.with_columns(pl.col("date_added").str.to_date("%Y-%m-%d"))
        return df

    @staticmethod
    def scrape_sp500_changes() -> pl.DataFrame:
        """
        Scrape S&P 500 component changes from Wikipedia and return as Polars dataframe.
        Returns a dataframe with columns: date, added, removed, reason
        """
        resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        soup = bs(resp.text, "html.parser")
        table = soup.find("table", id="changes")
        rows = []
        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            if len(cols) >= 4:
                date = cols[0].text.strip()
                added = cols[1].text.strip()
                name_added = cols[2].text.strip()
                removed = cols[3].text.strip()
                name_removed = cols[4].text.strip()
                rows.append(
                    {
                        "date": date,
                        "added": added,
                        "name_added": name_added,
                        "removed": removed,
                        "name_removed": name_removed,
                    }
                )
        df = pl.DataFrame(rows)
        df = df.with_columns(pl.col("date").str.strptime(pl.Date, "%B %d, %Y"))
        additions = df.filter(pl.col("added").str.strip_chars() != "").select(
            pl.col("date").alias("added"), pl.col("added").alias("tic"), pl.col("name_added")
        )
        removals = df.filter(pl.col("removed").str.strip_chars() != "").select(
            pl.col("date").alias("removed"), pl.col("removed").alias("tic"), pl.col("name_removed")
        )
        return additions, removals

    @staticmethod
    def scrape_sp500() -> pl.DataFrame:
        """
        List S&P500 stock info from wiki page and return Polars dataframe.
        Also includes addition and removal dates from changes table.
        """

        df = Scraper.scrape_sp500_constituents()
        changes_df = Scraper.scrape_sp500_changes()
        additions = changes_df.filter(pl.col("added").str.strip_chars() != "").select(
            pl.col("date").alias("addition_date"), pl.col("added").alias("tic")
        )
        removals = changes_df.filter(pl.col("removed").str.strip_chars() != "").select(
            pl.col("date").alias("removal_date"), pl.col("removed").alias("tic")
        )
        df = df.join(additions, on="tic", how="left")
        df = df.join(removals, on="tic", how="left")
        return df

    @staticmethod
    def get_exchange_stocks(exc_lis):
        headers = {
            "authority": "api.nasdaq.com",
            "accept": "application/json, text/plain, */*",
            "user-agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) ")
            + ("AppleWebKit/537.36 (KHTML, like Gecko)")
            + ("Chrome/87.0.4280.141 Safari/537.36"),
            "origin": "https://www.nasdaq.com",
            "sec-fetch-site": "same-site",
            "sec-fetch-mode": "cors",
            "sec-fetch-dest": "empty",
            "referer": "https://www.nasdaq.com/",
            "accept-language": "en-US,en;q=0.9",
        }
        stock_list = []
        for exc in exc_lis:
            params = (
                ("tableonly", "true"),
                ("exchange", exc),
                ("limit", "25"),
                ("offset", "0"),
                ("download", "true"),
            )
            r = requests.get(
                "https://api.nasdaq.com/api/screener/stocks",
                headers=headers,
                params=params,
            )
            data = r.json()["data"]
            df = pl.DataFrame(data["rows"])
            df = df.filter(pl.col("marketCap") != "0.00").select(["symbol", "name", "sector"])
            df = df.filter(~pl.col("symbol").str.contains(r"\.|\^"))
            stock_list.append(df)
        return pl.concat(stock_list).unique(subset=["symbol"])
