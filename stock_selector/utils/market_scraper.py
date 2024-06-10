#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:48:50 2020

@author: franciscosilva

"""

from datetime import datetime, timedelta
import time, requests, pandas, lxml
from lxml import html
import pandas as pd
import csv
import requests
from bs4 import BeautifulSoup as bs
import re
import json
import time
import numpy as np
import yfinance as yf
import time
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter

from config import get_config_dict


class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    pass


def get_session():
    
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
     hdrs = {"authority": "finance.yahoo.com",
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
                "user-agent": "Mozilla/5.0"}
     return hdrs


def scrape_page(url, header):
     page = requests.get(url, params=header)
     element_html = html.fromstring(page.content)
     table = element_html.xpath('//table')
     table_tree = lxml.etree.tostring(table[0], method='xml')
     panda = pandas.read_html(table_tree)
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


def get_splits(tic):
    # get stock splits
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
    Correct annualized dividend data, taking into account
    stock splits, in order to get correct cash dividend values.

    Parameters
    ----------
    df : Dataframe
        Data to correct
    symbol : str
        Firm ticker

    Returns
    -------
    DataFrame
        Split corrected dividend data
    """

    # get stock splits
    splits = get_splits(tic)
    df = data.copy()

    for date, row in splits.iterrows():
        n = float(row['Stock Splits'])
        df.loc[(df.datadate < row['Date']), col] /= n

    return df


def get_dividends(symbol, day_history=5900):
    """
    Retrieves split corrected dividend data from Yahoo Finance.

    Parameters
    ----------
    symbol : str
        Ticker of firm to retrieve dividends
    day_history : int
        Number of past days to retrieve

    Returns
    -------
    DataFrame
        Split corrected dividend data, with ex-dividend dates and respective cash dividend amounts
    """

    # Create datetime objects
    start = datetime.today() - timedelta(days=day_history)
    end = datetime.today()

    # Properly format the date to epoch time
    start = format_date(start)
    end = format_date(end)

    # Format the subdomains
    sub_div = subdomain_dividends(symbol, start, end)
    sub_sp = subdomain_splits(symbol, start, end)

    # Customize the request headers
    hdrs_div = header(sub_div)
    hdrs_sp = header(sub_sp)

    # Concatenate the subdomain with the base URL for dividend and split pages
    base_url = "https://finance.yahoo.com/quote/"
    url_div = base_url + sub_div
    url_sp = base_url + sub_sp

    # Retrieve data
    dividends = scrape_page(url_div, hdrs_div)

    # If there is no date column, Yahoo does not have the specified dividend data: return empty dataframe
    if ('Date' not in dividends[0]) or (dividends[0]['Date'][0] == 'No Dividends'):
        return pd.DataFrame()

    div = clean_dividends(symbol, dividends[0]).to_frame().reset_index()

    div['Date'] = pd.to_datetime(div['Date'])

    # Retrieve share splits
    splits = scrape_page(url_sp, hdrs_sp)

    # If there are no splits, return uncorrected dividend data
    if ('Date' not in splits[0]) or (splits[0]['Date'][0] == 'No Split'):
        div = div.rename(columns={symbol: "divq", 'Date': 'div_date'}, errors="raise")

        return div
    # Else, use split data to correct it
    else:
        sp = clean_splits(symbol, splits[0]).to_frame().reset_index()
        sp['Date'] = pd.to_datetime(sp['Date'])
        div = correct_splits(div,sp,symbol)
        div = div.rename(columns={symbol: "divq", 'Date': 'div_date'}, errors="raise")

        return div


def list_sp500_stocks():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})

    data = {"Symbol": [], "Sector": []}

    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.replace('\n', '')
        sector = row.findAll('td')[2].text
        data["Symbol"].append(ticker)
        data["Sector"].append(sector)

    return pd.DataFrame(data)


def get_market_data(tic, start, end):
    """
    Scrape daily market data for a stock.
    """

    time.sleep(1)
    session = get_session()
    
    t = yf.Ticker(tic, session=session)
    
    market_hist = t.history(start=start, 
                            end=end, 
                            auto_adjust=False).reset_index(drop=False)
    
    market_hist['Date'] = market_hist['Date'].apply(lambda x: x.date())
    market_hist['Date'] = pd.to_datetime(market_hist['Date'])
    
    if not len(market_hist):
        raise Exception("Empty DataFrame")
    
    return market_hist[['Date', 'Close', 'Adj Close', 'Volume']]


def get_earnings_dates(tic,  start, end):
    """
    Scrape earnings dates and eps estimates
    """
    
    n_quarters = int((end - start).days / 90) + 10

    time.sleep(1)
    session = get_session()
    t = yf.Ticker(tic, session=session)
    earn_dates = t.get_earnings_dates(limit=n_quarters).reset_index()
    earn_dates = earn_dates.rename(columns={"Earnings Date": "rdq",
                                            "EPS Estimate": "eps_est",
                                            "Reported EPS": "eps_rep",
                                            "Surprise(%)": "surprise_pct"
                                            })
    earn_dates['rdq'] = earn_dates['rdq'].apply(lambda x: x.date())
    earn_dates = earn_dates.drop_duplicates(subset=['rdq'])
    earn_dates['rdq'] = pd.to_datetime(earn_dates['rdq'])
    earn_dates.sort_values(by=['rdq'], inplace=True)
    earn_dates.dropna(subset=['eps_est', 'eps_rep', 'surprise_pct'], inplace=True)
    earn_dates = earn_dates[(earn_dates.rdq >= start) & 
                            (earn_dates.rdq < end)]
    
    if not len(earn_dates):
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
    fields_to_keep = get_config_dict("data")["macrotrends"]["income_statement"]
    
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
    
    # correct shares outstanding
    df = correct_splits(df, tic, 'cshoq')

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
    
    fields_to_keep = get_config_dict("data")["macrotrends"]["balance_sheet"]

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

    return df


def scrape_cash_flow_mt(tic, start, end):
    """
    Scraps quarterly Macrotrends CF data (fields can be changed by editing 'fields_to_keep' list)
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

    fields_to_keep = get_config_dict("data")["macrotrends"]["cash_flow"]
    
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
    
    return df


def scrape_ratios_mt(tic, start, end):
    """
    Scraps quarterly Macrotrends CF data (fields can be changed by editing 'fields_to_keep' list)
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

    fields_to_keep = get_config_dict("data")["macrotrends"]["ratios"]
    
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

    return df


def scrape_fundamental_data_yahoo(tic, start, end):
   
    time.sleep(1)
    session = get_session()
    
    t = yf.Ticker(tic, session=session)
    
    # Fields to preserve
    fields_to_keep = get_config_dict("data")["yahoo"]

    # retrieve 3 main documents
    is_df = t.quarterly_income_stmt.T
    bs_df = t.quarterly_balance_sheet.T
    cf_df = t.quarterly_cashflow.T
    
    # merge the data
    df = is_df.merge(bs_df, left_index=True, right_index=True, how='inner')
    df = df.merge(cf_df, left_index=True, right_index=True, how='inner')
    df = df.reset_index()
    
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
        
    if not len(df):
        raise Exception("Empty DataFrame")
    
    # correct shares outstanding
    #df = correct_splits(df, tic, 'cshoq')
    # get data in M$
    df[df.columns[1:]] /= 1000000
    df['dvq'] = -df['dvq']
    df['capxq'] = -df['capxq']
        
    return df
