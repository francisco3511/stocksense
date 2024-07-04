import lxml
import random
import re
import json
import time
import requests
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
        limiter=Limiter(RequestRate(2, Duration.SECOND*5)),  # max 1 request per 5 seconds
        bucket_class=MemoryQueueBucket,
        backend=SQLiteCache("yfinance.cache"),
    )
    
    session.headers['User-agent'] = 'my-program/1.0'
    #session.headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    return session


def format_date(date_datetime):
     date_timetuple = date_datetime.timetuple()
     date_mktime = time.mktime(date_timetuple)
     date_int = int(date_mktime)
     date_str = str(date_int)
     return date_str


def subdomain_dividends(symbol, start, end):
     format_url = "{0}/history?period1={1}&period2={2}"
     tail_url = "&interval=div%7Csplit&filter=div&frequency=1d"
     subdomain = format_url.format(symbol, start, end) + tail_url
     return subdomain


def subdomain_splits(symbol, start, end):
     format_url = "{0}/history?period1={1}&period2={2}"
     tail_url = "&interval=div%7Csplit&filter=split&frequency=1d"
     subdomain = format_url.format(symbol, start, end) + tail_url
     return subdomain


def header(subdomain):
     hdrs = {
        "authority": "finance.yahoo.com",
        "method": "GET",
        "path": subdomain,
        "scheme": "https",
        "accept": "text/html,application/xhtml+xml",
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "en-US,en;q=0.9",
        "cache-control": "no-cache",
        "cookie": "cookies",
        "dnt": "1",
        "pragma": "no-cache",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "same-origin",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1",
        "user-agent": "Mozilla/5.0"
    }
     return hdrs


def scrape_page(url, header):
     page = requests.get(url, params=header)
     element_html = lxml.html.fromstring(page.content)
     table = element_html.xpath('//table')
     table_tree = lxml.etree.tostring(table[0], method='xml')
     panda = pd.read_html(table_tree)
     return panda


def clean_dividends(symbol, dividends):
    index = len(dividends)
    # Drop last row
    dividends = dividends.drop(index-1)
    # Set date as series index
    dividends = dividends.set_index('Date')
    dividends = dividends['Dividends']
    dividends = dividends.str.replace(r'\Dividend', '')
    dividends.name = symbol
    # Force all values to be float and then drop NaNs
    dividends = dividends.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    dividends = dividends.dropna('')
    dividends = dividends.astype(float)
    # return parsed dividends
    return dividends


def clean_splits(symbol, splits):
     index = len(splits)
     splits = splits.drop(index-1)
     splits = splits.set_index('Date')
     splits = splits['Volume']
     splits.name = symbol
     return splits


def get_stock_splits(tic):
    """
    Get stock splits.
    """

    session = CachedLimiterSession(
        limiter=Limiter(RequestRate(2, Duration.SECOND*5)),  # max 2 requests per 5 seconds
        bucket_class=MemoryQueueBucket,
        backend=SQLiteCache("yfinance.cache"),
    )
    session.headers['User-agent'] = 'my-program/1.0'
    
    t = yf.Ticker(tic, session=session)
    
    splits = t.get_actions()
    splits = splits.loc[splits['Stock Splits'] > 0]
    splits = splits.reset_index()
    splits['Date'] = splits['Date'].apply(lambda x: x.date())
    splits['Date'] = pd.to_datetime(splits['Date'])


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


def scrape_sp500_stock_info():
    """
    List S&P500 stock info from wiki page and return pandas dataframe.
    """
    
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    time.sleep(random.uniform(1, 3))
    
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
    df['tic'] = df['tic'].str.replace('.','-')
    return df


def get_market_data(tic, start, end):
    """
    Scrape daily market data for a stock.
    """

    # create session
    session = get_session()
    
    # get data from yahoo finance
    t = yf.Ticker(tic, session=session)
    df = t.history(
        start=start, 
        end=end, 
        auto_adjust=False
    ).reset_index(drop=False)
    time.sleep(random.uniform(1, 3))
    
    # check if data was returned
    if df.empty:
        raise Exception("Empty DataFrame")
    
    # reformat data 
    df['Date'] = pd.to_datetime(df['Date'], format='ISO8601').dt.date
    df = df[['Date', 'Close', 'Adj Close', 'Volume']]
    df.columns = ['date', 'close', 'adj_close', 'volume']
    
    return df


def get_earnings_dates(tic,  start, end):
    """
    Scrape earnings dates and eps surprise.
    """
    
    n_quarters = int((end - start).days / 90) + 20

    session = get_session()
    t = yf.Ticker(tic, session=session)
    earn_dates = t.get_earnings_dates(limit=n_quarters).reset_index()
    time.sleep(random.uniform(1, 3))
    
    earn_dates = earn_dates.rename(
        columns={
            "Earnings Date": "rdq",
            "Surprise(%)": "surprise_pct"
        }
    )
    earn_dates['rdq'] = earn_dates['rdq'].apply(lambda x: x.date())
    earn_dates = earn_dates.drop_duplicates(subset=['rdq'])
    earn_dates['rdq'] = pd.to_datetime(earn_dates['rdq'])
    earn_dates.sort_values(by=['rdq'], inplace=True)
    earn_dates.dropna(subset=['surprise_pct'], inplace=True)
    earn_dates = earn_dates[
        (earn_dates.rdq >= start) & 
        (earn_dates.rdq < end)
    ]
    
    if earn_dates.empty:
        raise Exception("Empty DataFrame")
    
    return earn_dates

 
def scrape_income_statement_mt(tic, start, end):
    """
    Scraps quarterly Macrotrends income statement data (fields can be changed by editing 'fields_to_keep' list)
    """

    # create session
    session = get_session()
    url_base = 'https://www.macrotrends.net/stocks/charts/' + tic + '/x/income-statement?freq=Q'
    r_base = session.get(url_base)
    resp = r_base.url.split('/')
    url = 'https://www.macrotrends.net/stocks/charts/' + tic + '/' + resp[6] + '/income-statement?freq=Q'
    r = session.get(url)
    p = re.compile(r' var originalData = (.*?);\r\n\r\n\r', re.DOTALL)
    data = json.loads(p.findall(r.text)[0])

    headers = list(data[0].keys())
    headers.remove('popup_icon')
    result = []

    for row in data:
        soup = bs(row['field_name'], "lxml")
        field_name = soup.select_one('a, span').text
        fields = list(row.values())[2:]
        fields.insert(0, field_name)
        result.append(fields)

    df = pd.DataFrame(result, columns=headers)

    # Fields to preserve
    fields_to_keep = get_config("data")["macrotrends"]["income_statement"]
    
    df = df.T.reset_index()
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])

    # format data
    df = df[list(fields_to_keep.keys())]
    df = df.rename(columns=fields_to_keep)
    df['datadate'] = pd.to_datetime(df['datadate'])
    df = df[(df.datadate >= start) & (df.datadate < end)]
    
    if df.empty:
        raise Exception("Empty DataFrame")

    for col in df.columns[1:]:
        df.loc[df[col] == '', col] = None
        df[col] = df[col].astype(float)
    
    # correct shares outstanding
    df = correct_splits(df, tic, 'cshoq')
    
    time.sleep(random.uniform(1, 3))

    return df


def scrape_balance_sheet_mt(tic, start, end):
    """
    Scraps quarterly Macrotrends balance sheet data (fields can be changed by editing 'fields_to_keep' list)

    Parameters
    ----------
    tic : str
        Ticker of company for which data will be retrieved
    year_start : int
        Starting year for data records to retrieve

    Returns
    -------
    DataFrame
        Dataframe containing quarterly balance sheet data (identified by ending period of quarter)
    """

    # create session
    session = get_session()
    url_base = 'https://www.macrotrends.net/stocks/charts/' + tic + '/x/balance-sheet?freq=Q'
    r_base = session.get(url_base)
    resp = r_base.url.split('/')
    url = 'https://www.macrotrends.net/stocks/charts/' + tic + '/' + resp[6] + '/balance-sheet?freq=Q'
    r = session.get(url)
    p = re.compile(r' var originalData = (.*?);\r\n\r\n\r', re.DOTALL)
    data = json.loads(p.findall(r.text)[0])
    headers = list(data[0].keys())
    headers.remove('popup_icon')
    result = []

    for row in data:
        soup = bs(row['field_name'], "lxml")
        field_name = soup.select_one('a, span').text
        fields = list(row.values())[2:]
        fields.insert(0, field_name)
        result.append(fields)

    df = pd.DataFrame(result, columns=headers)
    
    fields_to_keep = get_config("data")["macrotrends"]["balance_sheet"]

    df = df.T.reset_index()
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    
    # format data
    df = df[list(fields_to_keep.keys())]
    df = df.rename(columns=fields_to_keep)
    df['datadate'] = pd.to_datetime(df['datadate'])
    df = df[(df.datadate >= start) & (df.datadate < end)]
    
    if len(df) == 0:
        raise Exception("Empty DataFrame")

    for col in df.columns[1:]:
        df.loc[df[col] == '', col] = None
        df[col] = df[col].astype(float)
        
    time.sleep(random.uniform(1, 3))

    return df


def scrape_cash_flow_mt(tic, start, end):
    """
    Scraps quarterly Macrotrends CF data.
    """

    # create session
    session = get_session()
    url_base = 'https://www.macrotrends.net/stocks/charts/' + tic + '/x/cash-flow-statement?freq=Q'
    r_base = session.get(url_base)
    resp = r_base.url.split('/')
    url = 'https://www.macrotrends.net/stocks/charts/' + tic + '/' + resp[6] + '/cash-flow-statement?freq=Q'
    r = session.get(url)
    p = re.compile(r' var originalData = (.*?);\r\n\r\n\r', re.DOTALL)
    data = json.loads(p.findall(r.text)[0])
    headers = list(data[0].keys())
    headers.remove('popup_icon')
    result = []

    for row in data:
        soup = bs(row['field_name'], "lxml")
        field_name = soup.select_one('a, span').text
        fields = list(row.values())[2:]
        fields.insert(0, field_name)
        result.append(fields)

    df = pd.DataFrame(result, columns=headers)

    fields_to_keep = get_config("data")["macrotrends"]["cash_flow"]
    
    df = df.T.reset_index()
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])

    # format data
    df = df[list(fields_to_keep.keys())]
    df = df.rename(columns=fields_to_keep)
    df['datadate'] = pd.to_datetime(df['datadate'])
    df = df[(df.datadate >= start) & (df.datadate < end)]
    
    if len(df) == 0:
        raise Exception("Empty DataFrame")

    for col in df.columns[1:]:
        df.loc[df[col] == '', col] = None
        df[col] = df[col].astype(float)
        
    time.sleep(random.uniform(1, 3))
    
    return df


def scrape_ratios_mt(tic, start, end):
    """
    Scraps quarterly Macrotrends financial ratio data.
    """

    # create session
    session = get_session()
    url_base = 'https://www.macrotrends.net/stocks/charts/' + tic + '/x/financial-ratios?freq=Q'
    r_base = session.get(url_base)
    resp = r_base.url.split('/')
    url = 'https://www.macrotrends.net/stocks/charts/' + tic + '/' + resp[6] + '/financial-ratios?freq=Q'
    r = session.get(url)
    p = re.compile(r' var originalData = (.*?);\r\n\r\n\r', re.DOTALL)
    data = json.loads(p.findall(r.text)[0])
    headers = list(data[0].keys())
    headers.remove('popup_icon')
    result = []

    for row in data:
        soup = bs(row['field_name'], "lxml")
        field_name = soup.select_one('a, span').text
        fields = list(row.values())[2:]
        fields.insert(0, field_name)
        result.append(fields)

    df = pd.DataFrame(result, columns=headers)

    fields_to_keep = get_config("data")["macrotrends"]["ratios"]
    
    df = df.T.reset_index()
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])

    df = df[list(fields_to_keep.keys())]
    df = df.rename(columns=fields_to_keep)
    df['datadate'] = pd.to_datetime(df['datadate'])
    df = df[(df.datadate >= start) & (df.datadate < end)]

    for col in df.columns[1:]:
        df.loc[df[col] == '', col] = None
        df[col] = df[col].astype(float)
        
    if not len(df):
        raise Exception("Empty DataFrame")
    
    time.sleep(random.uniform(1, 3))

    return df


def scrape_fundamental_data_macrotrends(tic, start, end):
    
    # scrape and merge 3 main documents
    df_is = scrape_income_statement_mt(tic, start, end)
    df_bs =  scrape_balance_sheet_mt(tic, start, end)
    df_cfs =  scrape_cash_flow_mt(tic, start, end)
    df_ratios = scrape_ratios_mt(tic, start, end)
                    
    df = pd.merge(df_is, df_bs, left_on='datadate', right_on='datadate', how='inner')
    df = pd.merge(df, df_cfs, left_on='datadate', right_on='datadate', how='inner')
    df = pd.merge(df, df_ratios, left_on='datadate', right_on='datadate', how='inner')
                
    # adjustments
    df['icaptq'] = (df['niq'] / df['icaptq']) * 100
    df['dvq'] = - df['dvq']
    
    return df


def scrape_fundamental_data_yahoo(ticker: str, start, end):
   
    session = get_session()
    
    t = yf.Ticker(ticker, session=session)
    
    # Fields to preserve
    fields_to_keep = get_config("data")["yahoo"]

    # retrieve 3 main documents
    is_df = t.quarterly_income_stmt.T
    bs_df = t.quarterly_balance_sheet.T
    cf_df = t.quarterly_cashflow.T
    
    # set timer
    time.sleep(random.uniform(1, 3))
    
    is_df.index = pd.to_datetime(is_df.index)
    bs_df.index = pd.to_datetime(bs_df.index)
    cf_df.index = pd.to_datetime(cf_df.index)
    
    if is_df.empty:
        raise Exception("Empty financials")
    
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
            
    df = df[list(fields_to_keep.keys())]
    df = df.rename(columns=fields_to_keep)
    df['datadate'] = pd.to_datetime(df['datadate'])
    df = df[(df.datadate >= start) & (df.datadate < end)]
    
    for col in df.columns[1:]:
        df.loc[df[col] == '', col] = None
        df[col] = df[col].astype(float)
    
    # data adjustments
    df[df.columns[1:]] /= 1000000
    df['dvq'] = -df['dvq']
    df['capxq'] = -df['capxq']
    df['tic'] = ticker
    df = df.drop_duplicates(subset=['datadate']).sort_values(by=['datadate'])        
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
        response = requests.get(url)
        time.sleep(random.uniform(1, 3))
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
            
        if data.empty:
            raise Exception("No insider data available")
        
        # data adjustments 
        df = pd.DataFrame(data, columns=field_names)
        df['tic'] = ticker
        df['filling_date'] = pd.to_datetime(df['filling_date'], format="ISO8601").dt.date
        df['trade_date'] = pd.to_datetime(df['trade_date'], format="ISO8601").dt.date
        df = df.sort_values(by='filling_date', ascending=True)
        return df
    except Exception as e:
        raise e


def get_stock_metadata(ticker: str) -> dict:
    """
    Retrieve 

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
    
    # query yahoo finance
    session = get_session()
    stock = yf.Ticker(ticker, session=session)
    metadata = stock.info
    
    # set sleeper
    time.sleep(random.uniform(1, 3))
    
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
    return metadata


def get_exchange_stocks(exc_lis):
    
    headers = {
        'authority': 'api.nasdaq.com',
        'accept': 'application/json, text/plain, */*',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36',
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
        r = requests.get('https://api.nasdaq.com/api/screener/stocks', headers=headers, params=params)
        data = r.json()['data']
        df = pd.DataFrame(data['rows'], columns=data['headers'])
        df = df.loc[df.marketCap != "0.00", ["symbol", "name", "sector"]]
        df = df[~df['symbol'].str.contains("\.|\^")]
        stock_list.append(df)
        time.sleep(1)
        
    return pd.concat(stock_list).drop_duplicates(subset=['symbol'])