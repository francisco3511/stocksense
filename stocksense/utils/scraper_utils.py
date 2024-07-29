import random
import time
import requests
import logging
import polars as pl
import numpy as np
import datetime as dt
import yfinance as yf
from bs4 import BeautifulSoup as bs
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter

from config import get_config

MAX_TIMER = 2

# Suppress logging from the yfinance and requests libraries
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    pass

def get_session():
    """
    Create session for yfinance queries.

    Returns
    -------
    CachedLimiterSession
        Session object.
    """
    session = CachedLimiterSession(
        # max 1 request per 5 seconds
        limiter=Limiter(RequestRate(2, Duration.SECOND*5)),
        bucket_class=MemoryQueueBucket,
        backend=SQLiteCache("yfinance.cache"),
    )
    session.headers['User-agent'] = 'my-program/1.0'
    return session

def get_stock_splits(tic):
    """
    Get stock splits.
    """
    session = get_session()
    t = yf.Ticker(tic, session=session)
    splits = t.get_actions()
    splits = splits[splits['Stock Splits'] > 0]
    splits = splits.reset_index()
    splits['Date'] = pl.from_pandas(splits)['Date'].str.to_date(format='%Y-%m-%d')
    return splits

def correct_splits(data, tic, col):
    """
    Correct market data, taking into account
    stock splits, in order to get accurate trackback.
    """
    # get stock splits
    splits = get_stock_splits(tic)
    df = data.clone()

    for date, row in splits.iterrows():
        n = float(row['Stock Splits'])
        df = df.with_columns(
            pl.when(pl.col('datadate') < row['Date']).then(pl.col(col) / n).otherwise(pl.col(col)).alias(col)
        )
    return df

def scrape_sp500_stock_info() -> pl.DataFrame:
    """
    List S&P500 stock info from wiki page and return Polars dataframe.
    """
    resp = requests.get(
        'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    )
    soup = bs(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})

    data = {
        "tic": [],
        "name": [],
        "sector": [],
    }

    for row in table.findAll('tr')[1:]:
        # find entries
        ticker = row.findAll('td')[0].text.replace('\n', '')
        security = row.findAll('td')[1].text.replace('\n', '')
        sector = row.findAll('td')[2].text.replace('\n', '')
        # save data
        data["tic"].append(ticker)
        data["name"].append(security)
        data["sector"].append(sector)

    df = pl.DataFrame(data)
    df = df.with_columns(pl.col('tic').str.replace(".", "-", literal=True))
    return df

def scrape_fundamental_data_macrotrends(ticker: str, start: dt.date, end: dt.date) -> None:
    raise Exception("Not implemented.")

def scrape_fundamental_data_yahoo(ticker: str, start: dt.date, end: dt.date) -> pl.DataFrame:
    """
    Scraps fundamental data from Yahoo Finance using yfinance lib, searching
    for financial records released between two dates.

    Parameters
    ----------
    ticker : str
        Target stock ticker.
    start : dt.date
        Starting date.
    end : dt.date
        Ending date

    Returns
    -------
    pl.DataFrame
        Financial report data table.

    Raises
    ------
    Exception
        If no financial records are available.
    """
    session = get_session()
    t = yf.Ticker(ticker, session=session)

    # Fields to preserve
    fields_to_keep = get_config("data")["yahoo"]

    # retrieve 3 main documents
    is_df = pl.from_pandas(t.quarterly_income_stmt.T.reset_index())
    bs_df = pl.from_pandas(t.quarterly_balance_sheet.T.reset_index())
    cf_df = pl.from_pandas(t.quarterly_cashflow.T.reset_index())

    # set timer
    time.sleep(random.uniform(1, MAX_TIMER))

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
        (pl.col('datadate') >= start) & 
        (pl.col('datadate') < end
    ))

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
    df = df.with_columns(pl.lit(ticker).alias('tic'))
    return df

def get_market_data(ticker, start):
    """
    Scrape daily market data for a stock, until present.
    """
    # create session and get data
    session = get_session()
    t = yf.Ticker(ticker, session=session)

    # get historical data
    df = pl.from_pandas(t.history(
        start=start,
        auto_adjust=False
    ).reset_index(drop=False))

    # set timer
    time.sleep(random.uniform(1, MAX_TIMER))

    # check if data was returned
    if df.is_empty():
        raise Exception("Empty DataFrame")

    # reformat data
    df = df.with_columns([
        pl.col('Date').dt.date().alias('date'),
        pl.col('Close').alias('close'),
        pl.col('Adj Close').alias('adj_close'),
        pl.col('Volume').alias('volume')
    ])
    df = df.select(['date', 'close', 'adj_close', 'volume']).with_columns(pl.lit(ticker).alias('tic'))
    return df

def get_earnings_dates(tic: str, start: dt.date, end: dt.date):
    """
    Scrape earnings dates and eps surprise.
    """
    n_quarters = int((end - start).days / 90) + 20

    session = get_session()
    t = yf.Ticker(tic, session=session)
    df = pl.from_pandas(t.get_earnings_dates(limit=n_quarters).reset_index())
    time.sleep(random.uniform(1, MAX_TIMER))

    df = df.rename({
        "Earnings Date": "rdq",
        "Surprise(%)": "surprise_pct"
    })

    # format dates and filter data
    df = df.select(['rdq', 'surprise_pct'])
    df = df.with_columns(pl.col('rdq').dt.date())
    df = df.filter((pl.col('rdq') >= start) & (pl.col('rdq') < end))
    df = df.unique(subset=['rdq']).sort('rdq').drop_nulls(subset=['surprise_pct', 'rdq'])

    if df.is_empty():
        raise Exception("Empty DataFrame")

    return df

def get_financial_data(ticker: str, start: dt.date, end: dt.date, method: str = "yahoo") -> pl.DataFrame:
    if method == "yahoo":
        df = scrape_fundamental_data_yahoo(ticker, start, end)
    else:
        df = scrape_fundamental_data_macrotrends(ticker, start, end)

    # get earnings dates and estimates
    earn_dates = get_earnings_dates(ticker, start, end)

    # ensure correct date typing for merge asof (only datetime supported)
    df = df.with_columns(pl.col('datadate').dt.date()).sort('datadate')
    earn_dates = earn_dates.with_columns(pl.col('rdq').dt.date()).sort('rdq')

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
    df = df.with_columns(pl.when(pl.col('rdq').is_null()).then(today).otherwise(pl.col('rdq')).alias('rdq'))

    return df

def get_stock_insider_data(ticker: str) -> pl.DataFrame:
    """
    Scrape OpenInsider.com insider trading data for a given stock.

    Parameters
    ----------
    ticker : str
        Target stock ticker.

    Returns
    -------
    pl.DataFrame
        Insider trading data table.

    Raises
    ------
    ValueError
        When scraping process fails.
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
        f'http://openinsider.com/screener?s={ticker}&o=&pl=&ph=&ll=&lh=&fd=0'
        '&fdr=&td=0&tdr=&fdlyl=&fdlyh=&daysago=&xp=1&xs=1&vl=&vh=&ocl=&och=&'
        'sic1=-1&sicl=100&sich=9999&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&'
        'v2h=&oc2l=&oc2h=&sortcol=0&cnt=1000&page=1'
    )
    try:
        data = []

        # get insider trading data
        response = requests.get(url)

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
        df = df.with_columns(pl.lit(ticker).alias('tic'))

        if df.is_empty():
            raise Exception("No insider data available")

        # format dates and sort
        df = df.with_columns([
            pl.col('filling_date').str.to_datetime("%Y-%m-%d %H:%M:%S").dt.date(),
            pl.col('trade_date').str.to_date("%Y-%m-%d")
        ])

        df = df.sort('filling_date')
        return df
    except Exception as e:
        raise e

def get_stock_info(ticker: str) -> dict:
    """
    Retrieve stock ancillary information.

    Parameters
    ----------
    ticker : str
        Target stock ticker.

    Returns
    -------
    dict
        Dictionary with target info fields.

    Raises
    ------
    Exception
        When no data was found.
    """
    # query yahoo finance and get info
    session = get_session()
    stock = yf.Ticker(ticker, session=session)
    data = stock.info

    # set sleeper
    time.sleep(random.uniform(1, MAX_TIMER))

    # verify data and format it
    if not data:
        raise Exception("Empty stock info.")

    fields = get_config("data")["yahoo_info"]

    record = dict.fromkeys(list(fields.keys()), None)
    record['tic'] = ticker

    for yh_key, key in fields.items():
        if yh_key in data:
            record[key] = data[yh_key]
    
    return record

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
        time.sleep(1)

    return pl.concat(stock_list).unique(subset=['symbol'])
