import time
import requests
import logging
import polars as pl
import datetime as dt
import yfinance as yf
from bs4 import BeautifulSoup as bs
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter
from loguru import logger

from config import get_config

# Suppress logging from the yfinance and requests libraries
logging.getLogger('yfinance').setLevel(logging.CRITICAL)


class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    pass


class Scraper:
    """
    Stock web scraping class (inc. financial, market, info and insider info)
    Currently supports yfinance source only.
    """
    def __init__(self, tic, source):
        self.tic = tic
        self.source = source
        self.session = self._get_session()
        if self.source == "yfinance":
            self.handler = self._get_yfinance_handler()

    def _get_session(self):
        """
        Create session for yfinance queries.

        :return CachedLimiterSession: Session object.
        """
        session = CachedLimiterSession(
            limiter=Limiter(RequestRate(2, Duration.SECOND * 5)),
            bucket_class=MemoryQueueBucket,
            backend=SQLiteCache(f"data/cache/{self.source}.cache"),
        )
        session.headers['User-agent'] = 'my-program/1.0'
        return session

    def _get_yfinance_handler(self):
        """
        Get yfinance stock ticker handler.

        :return yf.Ticker: stock ticker handler
        """
        return yf.Ticker(self.tic, session=self.session)

    def _get_market_data_yfinance(self, start_date):
        """
        Scrape daily market data for a stock, until present.
        """
        start = time.time()
        df = pl.from_pandas(self.handler.history(
            start=start_date,
            auto_adjust=False
        ).reset_index(drop=False))
        end = time.time()

        if df.is_empty():
            raise Exception("Empty DataFrame")

        df = df.with_columns([
            pl.col('Date').dt.date().alias('date'),
            pl.col('Close').alias('close'),
            pl.col('Adj Close').alias('adj_close'),
            pl.col('Volume').alias('volume'),
            pl.lit(self.tic).alias('tic')
        ])
        df = df.select(['date', 'close', 'adj_close', 'volume', 'tic'])
        logger.info(f"scraping took {end - start:.2f} seconds")
        return df

    def _get_stock_info_yfinance(self) -> dict:
        """
        Scrape current info using yfinance.

        :raises Exception: no info available.
        :return dict: current stock info.
        """
        start = time.time()
        data = self.handler.info
        end = time.time()

        if not data:
            raise Exception("Empty stock info.")

        fields = get_config("data")["yahoo_info"]
        record = dict.fromkeys(list(fields.values()), None)
        record['tic'] = self.tic

        for yh_key, key in fields.items():
            if yh_key in data:
                record[key] = data[yh_key]

        logger.info(f"scraping took {end - start:.2f} seconds")
        return record

    def _get_fundamental_data_yfinance(
        self,
        start_date: dt.date,
        end_date: dt.date
    ) -> pl.DataFrame:
        """
        Scraps fundamental data from Yahoo Finance using yfinance lib, searching
        for financial records released between two dates.

        :param dt.date start_date: starting date.
        :param dt.date end_date: ending date
        :raises Exception: no financial records are available.
        :return pl.DataFrame: financial report data from yfinance.
        """
        start = time.time()

        # fields to preserve
        fields_to_keep = get_config("data")["yahoo"]

        # retrieve 3 main documents
        is_df = pl.from_pandas(self.handler.quarterly_income_stmt.T.reset_index())
        bs_df = pl.from_pandas(self.handler.quarterly_balance_sheet.T.reset_index())
        cf_df = pl.from_pandas(self.handler.quarterly_cashflow.T.reset_index())
        end = time.time()

        # parse dates
        is_df = is_df.with_columns(
            pl.col('index').dt.date().alias('index')
        ).sort('index')
        bs_df = bs_df.with_columns(
            pl.col('index').dt.date().alias('index')
        ).sort('index')
        cf_df = cf_df.with_columns(
            pl.col('index').dt.date().alias('index')
        ).sort('index')

        # merge the data based on dates
        df = is_df.join_asof(
            bs_df,
            on='index',
            strategy='forward',
            tolerance=dt.timedelta(days=40)
        ).join_asof(
            cf_df,
            on='index',
            strategy='forward',
            tolerance=dt.timedelta(days=40)
        )

        # validate columns and create placeholders
        for c in list(fields_to_keep.keys()):
            if c not in df.columns:
                df = df.with_columns([
                    pl.lit(None).alias(c),
                ])

        # filter columns
        df = df.select(list(fields_to_keep.keys()))
        df = df.rename(fields_to_keep)
        df = df.filter(
            (pl.col('datadate') >= start_date) &
            (pl.col('datadate') < end_date)
        )

        if df.is_empty():
            raise Exception("Empty financials")

        for c in df.columns[1:]:
            df = df.with_columns(
                pl.col(c).cast(pl.Float64)
            )
            df = df.with_columns((pl.col(c) / 1000000).round(3).alias(c))

        # data corrections
        df = df.with_columns([
            (-pl.col('dvq')).alias('dvq'),
            (-pl.col('capxq')).alias('capxq')
        ])
        df = df.unique(subset=['datadate']).sort('datadate')
        df = df.with_columns(pl.lit(self.tic).alias('tic'))
        logger.info(f"scraping took {end - start:.2f} seconds")
        return df

    def _get_earnings_dates_yfinance(self, start_date: dt.date, end_date: dt.date):
        """
        Scrape earnings dates and eps surprise.

        :param dt.date start_date: starting date.
        :param dt.date end_date: ending date.
        :raises Exception: empty data.
        :return pl.DataFrame: _description_
        """
        n_quarters = int((end_date - start_date).days / 90) + 20

        start = time.time()
        df = pl.from_pandas(
            self.handler.get_earnings_dates(limit=n_quarters).reset_index()
        )
        end = time.time()

        df = df.rename({
            "Earnings Date": "rdq",
            "Surprise(%)": "surprise_pct"
        })

        # format dates and filter data
        df = df.select(['rdq', 'surprise_pct'])
        df = df.with_columns(pl.col('rdq').dt.date())
        df = df.filter((pl.col('rdq') >= start_date) & (pl.col('rdq') < end_date))
        df = df.unique(subset=['rdq']).sort('rdq').drop_nulls(subset=['surprise_pct', 'rdq'])

        if df.is_empty():
            raise Exception("Empty DataFrame")

        logger.info(f"scraping took {end - start:.2f} seconds")
        return df

    def get_market_data(self, start_date):
        """
        Scrape market data.

        :raises Exception: scraping method not implemented.
        :return _type_: scraped market data.
        """
        if self.source == "yfinance":
            # scrape market data from yfinance
            return self._get_market_data_yfinance(start_date)
        else:
            raise Exception("Other methods not implemented")

    def get_stock_info(self):
        """
        Scrape current stock info.

        :raises Exception: scraping method not implemented.
        :return _type_: scraped stock info.
        """
        if self.source == "yfinance":
            # scrape stock current info from yfinance
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

        :param dt.date start_date: starting date.
        :param dt.date end_date: ending date.
        :raises Exception: scraping method not implemented.
        :return pl.DataFrame: financial data.
        """
        if self.source == "yfinance":
            # scrape fundamental data from yfinance
            df = self._get_fundamental_data_yfinance(start_date, end_date)

            # get earnings dates and estimates
            earn_dates = self._get_earnings_dates_yfinance(start_date, end_date)

            df = df.with_columns(pl.col('datadate').dt.date()).sort('datadate')
            earn_dates = earn_dates.with_columns(pl.col('rdq').dt.date()).sort('rdq')
        else:
            raise Exception("Other methods not implemented")

        # merge data
        df = df.join_asof(
            earn_dates,
            left_on='datadate',
            right_on='rdq',
            strategy='forward',
            tolerance=dt.timedelta(days=80)
        )

        # in cases where data is found but no release dt, defer to today
        today = dt.datetime.now().date()
        df = df.with_columns(pl.when(
            pl.col('rdq').is_null()
            ).then(today).otherwise(pl.col('rdq')).alias('rdq')
        )
        return df

    def get_stock_insider_data(self) -> pl.DataFrame:
        """
        Scrape OpenInsider.com insider trading data for a given stock.

        :raises ValueError: when scraping process fails.
        :return pl.DataFrame: insider trading data table.
        """
        # select fields to retrieve
        field_names = [
            'filling_date',
            'trade_date',
            'owner_name',
            'title',
            'transaction_type',
            'last_price',
            'qty',
            'shares_held',
            'owned',
            'value'
        ]
        # set url format for OpenInsider.com
        url = (
            f'http://openinsider.com/screener?s={self.tic}&o=&pl=&ph=&ll=&lh=&fd=0'
            '&fdr=&td=0&tdr=&fdlyl=&fdlyh=&daysago=&xp=1&xs=1&vl=&vh=&ocl=&och=&'
            'sic1=-1&sicl=100&sich=9999&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&'
            'v2h=&oc2l=&oc2h=&sortcol=0&cnt=1000&page=1'
        )
        try:
            data = []

            # get insider trading data
            start = time.time()
            response = requests.get(url)
            end = time.time()

            # process values
            soup = bs(response.text, 'html.parser')
            table = soup.find('table', {'class': 'tinytable'})
            if not table:
                raise ValueError("Table not found on page")

            rows = table.find_all('tr')
            for row in rows[1:]:
                cols = row.find_all('td')
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

            # data adjustments
            df = pl.DataFrame(data, schema=field_names, orient='row')
            df = df.with_columns(pl.lit(self.tic).alias('tic'))

            if df.is_empty():
                raise Exception("No insider data available")

            # format dates and sort
            df = df.with_columns([
                pl.col('filling_date').str.to_datetime("%Y-%m-%d %H:%M:%S").dt.date(),
                pl.col('trade_date').str.to_date("%Y-%m-%d")
            ])
            df = df.sort('filling_date')

            logger.info(f"scraping took {end - start:.2f} seconds")
            return df
        except Exception as e:
            raise e

    @staticmethod
    def scrape_sp500_stock_info() -> pl.DataFrame:
        """
        List S&P500 stock info from wiki page and return Polars dataframe.
        """
        start = time.time()
        resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        end = time.time()
        soup = bs(resp.text, 'lxml')
        table = soup.find('table', {'class': 'wikitable sortable'})

        data = {"tic": [], "name": [], "sector": []}

        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[0].text.replace('\n', '')
            security = row.findAll('td')[1].text.replace('\n', '')
            sector = row.findAll('td')[2].text.replace('\n', '')
            data["tic"].append(ticker)
            data["name"].append(security)
            data["sector"].append(sector)

        df = pl.DataFrame(data)
        df = df.with_columns(pl.col('tic').str.replace(".", "-", literal=True))
        logger.info(f"scraping took {end - start:.2f} seconds")
        return df

    @staticmethod
    def get_exchange_stocks(exc_lis):
        headers = {
            'authority': 'api.nasdaq.com',
            'accept': 'application/json, text/plain, */*',
            'user-agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) ') +
            ('AppleWebKit/537.36 (KHTML, like Gecko)') +
            ('Chrome/87.0.4280.141 Safari/537.36'),
            'origin': 'https://www.nasdaq.com',
            'sec-fetch-site': 'same-site',
            'sec-fetch-mode': 'cors',
            'sec-fetch-dest': 'empty',
            'referer': 'https://www.nasdaq.com/',
            'accept-language': 'en-US,en;q=0.9',
        }
        stock_list = []
        for exc in exc_lis:
            params = (
                ('tableonly', 'true'),
                ('exchange', exc),
                ('limit', '25'),
                ('offset', '0'),
                ('download', 'true'),
            )
            r = requests.get(
                'https://api.nasdaq.com/api/screener/stocks',
                headers=headers,
                params=params
            )
            data = r.json()['data']
            df = pl.DataFrame(data['rows'])
            df = df.filter(pl.col('marketCap') != "0.00").select(["symbol", "name", "sector"])
            df = df.filter(~pl.col('symbol').str.contains(r"\.|\^"))
            stock_list.append(df)
        return pl.concat(stock_list).unique(subset=['symbol'])
