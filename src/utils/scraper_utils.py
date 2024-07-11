import random
import time
import requests
import logging
import pandas as pd
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
    splits = splits.loc[splits['Stock Splits'] > 0]
    splits = splits.reset_index()
    splits['Date'] = pd.to_datetime(splits['Date'], format='ISO8601').dt.date
    return splits


def correct_splits(data, tic, col):
    """
    Correct market data, taking into account
    stock splits, in order to get accurate trackback.
    """

    # get stock splits
    splits = get_stock_splits(tic)
    df = data.copy()

    for date, row in splits.iterrows():
        n = float(row['Stock Splits'])
        df.loc[(df.datadate < row['Date']), col] /= n

    return df


def scrape_sp500_stock_info() -> pd.DataFrame:
    """
    List S&P500 stock info from wiki page and return pandas dataframe.
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

    df = pd.DataFrame(data)
    df['tic'] = df['tic'].str.replace('.', '-')
    return df


def scrape_fundamental_data_macrotrends(
    ticker: str,
    start: dt.date,
    end: dt.date
) -> None:
    raise Exception("Not implemented.")


def scrape_fundamental_data_yahoo(
    ticker: str,
    start: dt.date,
    end: dt.date
) -> pd.DataFrame:
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
    pd.DataFrame
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
    is_df = t.quarterly_income_stmt.T
    bs_df = t.quarterly_balance_sheet.T
    cf_df = t.quarterly_cashflow.T

    # set timer
    time.sleep(random.uniform(1, MAX_TIMER))

    is_df.index = pd.to_datetime(is_df.index)
    bs_df.index = pd.to_datetime(bs_df.index)
    cf_df.index = pd.to_datetime(cf_df.index)

    # merge the data
    df = pd.merge_asof(
        is_df.sort_index(),
        bs_df.sort_index(),
        left_index=True,
        right_index=True,
        direction='forward',
        tolerance=dt.timedelta(days=40)
    )
    df = pd.merge_asof(
        df.sort_index(),
        cf_df.sort_index(),
        left_index=True,
        right_index=True,
        direction='forward',
        tolerance=dt.timedelta(days=40)
    ).reset_index()

    # validate columns and create placeholders
    for c in list(fields_to_keep.keys()):
        if c not in df.columns:
            df[c] = np.nan

    # filter columns
    df = df[list(fields_to_keep.keys())]

    df = df.rename(columns=fields_to_keep)
    df['datadate'] = pd.to_datetime(df['datadate'], format='ISO8601').dt.date
    df = df[(df.datadate >= start) & (df.datadate < end)]

    if is_df.empty:
        raise Exception("Empty financials")

    for col in df.columns[1:]:
        df.loc[df[col] == '', col] = None
        df[col] = df[col].astype(float)

    # data corrections
    df[df.columns[1:]] /= 1000000
    df['dvq'] = - df['dvq']
    df['capxq'] = - df['capxq']
    df = df.round(3)
    df = df.drop_duplicates(subset=['datadate']).sort_values(by=['datadate'])
    df['tic'] = ticker
    return df


def get_market_data(ticker, start):
    """
    Scrape daily market data for a stock, until present.
    """

    # create session and get data
    session = get_session()
    t = yf.Ticker(ticker, session=session)
    df = t.history(
        start=start,
        auto_adjust=False
    ).reset_index(drop=False)

    # set timer
    time.sleep(random.uniform(1, MAX_TIMER))

    # check if data was returned
    if df.empty:
        raise Exception("Empty DataFrame")

    # reformat data
    df['Date'] = pd.to_datetime(df['Date'], format='ISO8601').dt.date
    df = df[['Date', 'Close', 'Adj Close', 'Volume']]
    df.columns = ['date', 'close', 'adj_close', 'volume']
    df['tic'] = ticker
    return df


def get_earnings_dates(tic: str,  start: dt.date, end: dt.date):
    """
    Scrape earnings dates and eps surprise.
    """

    n_quarters = int((end - start).days / 90) + 20

    session = get_session()
    t = yf.Ticker(tic, session=session)
    df = t.get_earnings_dates(limit=n_quarters).reset_index()
    time.sleep(random.uniform(1, MAX_TIMER))

    df = df.rename(columns={
        "Earnings Date": "rdq",
        "Surprise(%)": "surprise_pct"
    })

    # format dates and filter data
    df = df[['rdq', 'surprise_pct']]
    df['rdq'] = pd.to_datetime(df['rdq'], format='ISO8601').dt.date
    df = df[(df.rdq >= start) & (df.rdq < end)]
    df = df.drop_duplicates(subset=['rdq'])
    df.sort_values(by=['rdq'], inplace=True)
    df.dropna(subset=['surprise_pct'], inplace=True)
    df.dropna(subset=['rdq'], inplace=True)

    if df.empty:
        raise Exception("Empty DataFrame")

    return df


def get_financial_data(
    ticker: str,
    start: dt.date,
    end: dt.date,
    method: str = "yahoo"
) -> pd.DataFrame:

    if method == "yahoo":
        df = scrape_fundamental_data_yahoo(ticker, start, end)
    else:
        df = scrape_fundamental_data_macrotrends(ticker, start, end)

    # get earnings dates and estimates
    earn_dates = get_earnings_dates(ticker, start, end)

    # ensure correct date typing for merge asof (only datetime supported)
    df['datadate'] = pd.to_datetime(df['datadate'], format="ISO8601")
    earn_dates['rdq'] = pd.to_datetime(earn_dates['rdq'], format="ISO8601")

    # merge data
    df = pd.merge_asof(
        df,
        earn_dates,
        left_on='datadate',
        right_on='rdq',
        direction='forward',
        tolerance=dt.timedelta(days=80)
    )

    # ensure correct date typing
    df['datadate'] = pd.to_datetime(df['datadate'], format="ISO8601").dt.date
    df['rdq'] = pd.to_datetime(df['rdq'], format="ISO8601").dt.date
    
    # in cases where data is found but no release dt, defer to today
    today = dt.datetime.now().date()
    df['rdq'] = df['rdq'].fillna(today)

    return df


def get_stock_insider_data(ticker: str) -> pd.DataFrame:
    """
    Scrape OpenInsider.com insider trading data for a given stock.

    Parameters
    ----------
    ticker : str
        Target stock ticker.

    Returns
    -------
    pd.DataFrame
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
        df = pd.DataFrame(data, columns=field_names)
        df['tic'] = ticker

        if df.empty:
            raise Exception("No insider data available")

        # format dates and sort
        df['filling_date'] = pd.to_datetime(
            df['filling_date'], format="ISO8601"
        ).dt.date
        df['trade_date'] = pd.to_datetime(
            df['trade_date'], format="ISO8601"
        ).dt.date
        df = df.sort_values(by='filling_date', ascending=True)
        return df
    except Exception as e:
        raise e


def get_stock_metadata(ticker: str) -> dict:
    """
    Retrieve stock metadata.

    Parameters
    ----------
    ticker : str
        Target stock ticker.

    Returns
    -------
    dict
        Dictionary with target metadata fields.

    Raises
    ------
    Exception
        When no data was found.
    """

    # query yahoo finance and get metadata
    session = get_session()
    stock = yf.Ticker(ticker, session=session)
    metadata = stock.info

    # set sleeper
    time.sleep(random.uniform(1, MAX_TIMER))

    # verify data and format it
    if not metadata:
        raise Exception("Empty stock info.")

    record = {
        'tic': ticker,
        'shares_outstanding': None,
        'enterprise_value': None,
        'rec_key': None,
        'forward_pe': None
    }

    # check if retrieved metadata contains db fields
    if 'sharesOutstanding' in metadata:
        record['shares_outstanding'] = metadata['sharesOutstanding']
    if 'enterpriseValue' in metadata:
        record['enterprise_value'] = metadata['enterpriseValue']
    if 'recommendationKey' in metadata:
        record['rec_key'] = metadata['recommendationKey']
    if 'forwardPE' in metadata:
        record['forward_pe'] = metadata['forwardPE']
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
        df = pd.DataFrame(data['rows'], columns=data['headers'])
        df = df.loc[df.marketCap != "0.00", ["symbol", "name", "sector"]]
        df = df[~df['symbol'].str.contains(r"\.|\^")]
        stock_list.append(df)
        time.sleep(1)

    return pd.concat(stock_list).drop_duplicates(subset=['symbol'])
