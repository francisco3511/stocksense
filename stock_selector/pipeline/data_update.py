"""
@author: franciscosilva
@email: francisco3597@gmail.com
"""

import requests
import pickle
from bs4 import BeautifulSoup as bs
import re
import json
import time
import pandas as pd
import numpy as np
import urllib.request as ur
from yahooquery import Ticker
import stock_selector.utils.market_scraper as ms
import ratio_generator as rg

pd.options.display.float_format = '{:.4f}'.format
pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.max_columns = None
pd.options.display.max_rows = None

# Black list: these companies have no records in Macrotrends DB
ignore_list = ['AET', 'AVP', 'JEC', 'JCP', 'RTN', 'STI', 'ANDV', 'UTX', 'S', 'CBS', 'UNP',
                'XL', 'FII', 'AGN', 'AKS', 'CVG', 'FTR', 'COL', 'LM', 'DO', 'CTL', 'DNR', 'MYL', 'NBL', 'TIF', 'AIV', 
               'ETFC', 'CXO']

def scrape_quarter_income_statement(tic, year_start=2018):
    """
    Scraps quarterly Macrotrends income statement data (fields can be changed by editing 'fields_to_keep' list)

    Parameters
    ----------
    tic : str
        Ticker of company for which data will be retrieved
    year_start : int
        Starting year for data records to retrieve

    Returns
    -------
    DataFrame
        Dataframe containing quarterly income statement data (identified by ending period of quarter)
    """

    url_base = 'https://www.macrotrends.net/stocks/charts/' + tic + '/x/income-statement?freq=Q'
    r_base = requests.get(url_base)
    resp = r_base.url.split('/')
    url = 'https://www.macrotrends.net/stocks/charts/' + tic + '/' + resp[6] + '/income-statement?freq=Q'
    r = requests.get(url)

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
    fields_to_keep = ['field_name',
                      'Revenue', 'Gross Profit', 'SG&A Expenses',
                      'Net Income', 'EBITDA', 'Shares Outstanding']

    df = df.T.reset_index()
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])

    df = df[fields_to_keep]
    df = df.rename(columns={'field_name': 'datadate'})
    df['datadate'] = pd.to_datetime(df['datadate'])
    df = df[df.datadate.dt.year >= year_start]

    for col in df.columns[1:]:
        df[col][df[col] == ''] = None
        df[col] = df[col].astype(float)

    return df

def scrape_quarter_balance_sheet(tic, year_start=2018):
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

    url_base = 'https://www.macrotrends.net/stocks/charts/' + tic + '/x/balance-sheet?freq=Q'
    r_base = requests.get(url_base)
    resp = r_base.url.split('/')
    url = 'https://www.macrotrends.net/stocks/charts/' + tic + '/' + resp[6] + '/balance-sheet?freq=Q'
    r = requests.get(url)

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

    fields_to_keep = ['field_name', # Keep this field
                      'Total Current Assets', 'Total Assets', 'Cash On Hand',
                      'Receivables', 'Inventory', 'Property, Plant, And Equipment',
                      'Total Current Liabilities', 'Long Term Debt', 'Total Liabilities',
                      'Retained Earnings (Accumulated Deficit)', 'Share Holder Equity']

    df = df.T.reset_index()
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])

    df = df[fields_to_keep]
    df = df.rename(columns={'field_name': 'datadate'})
    df['datadate'] = pd.to_datetime(df['datadate'])
    df = df[df.datadate.dt.year >= year_start]

    for col in df.columns[1:]:
        df[col][df[col] == ''] = None
        df[col] = df[col].astype(float)

    return df

def scrape_cf_yahoo(tic, year_start=2018):
    """
    Scraps quarterly Yahoo Finance cash flow Statement data (fields can be changed by editing 'fields_to_keep' list)
    This data can be limited as neither Macrotrends or Yahoo allow for access to quarterly CFS of more than 4 quarters
    into the past.

    Parameters
    ----------
    tic : str
        Ticker of company for which data will be retrieved
    year_start : int
        Starting year for data records to retrieve

    Returns
    -------
    DataFrame
        Dataframe containing quarterly cash flow Statement data (identified by ending period of quarter)
    """

    while True:
        t = Ticker(tic)
        df = t.cash_flow('q')
        if isinstance(df, pd.DataFrame):
            break

    fields_to_keep = ['asOfDate', # Keep this field
                      'OperatingCashFlow', 'InvestingCashFlow', 'FreeCashFlow']

    df = df[fields_to_keep]
    df = df.rename(columns={'asOfDate': 'datadate'})
    df['datadate'] = pd.to_datetime(df['datadate'])
    df = df[df.datadate.dt.year >= year_start]
    for col in df.columns[1:]:
        df[col][df[col] == ''] = None
        df[col] = df[col].astype(float)

    df['OperatingCashFlow'] = df['OperatingCashFlow'] / 1000000
    df['InvestingCashFlow'] = df['InvestingCashFlow'] / 1000000
    df['FreeCashFlow'] = df['FreeCashFlow'] / 1000000

    return df

def scrape_quarter_ratios(tic, year_start=2018):
    """
    Scraps other quarterly Macrotrends data items (fields can be changed by editing 'fields_to_keep' list)

    Parameters
    ----------
    tic : str
        Ticker of company for which data will be retrieved
    year_start : int
        Starting year for data records to retrieve

    Returns
    -------
    DataFrame
        Dataframe containing quarterly item data (identified by ending period of quarter)
    """

    url_base = 'https://www.macrotrends.net/stocks/charts/' + tic + '/x/financial-ratios?freq=Q'
    r_base = requests.get(url_base)
    resp = r_base.url.split('/')
    url = 'https://www.macrotrends.net/stocks/charts/' + tic + '/' + resp[6] + '/financial-ratios?freq=Q'
    r = requests.get(url)

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
    fields_to_keep = ['field_name', # Keep this field
                      'ROI - Return On Investment']
    df = df.T.reset_index()
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])

    df = df[fields_to_keep]
    df = df.rename(columns={'field_name': 'datadate',
                            'ROI - Return On Investment': 'roiq'})
    df['datadate'] = pd.to_datetime(df['datadate'])
    df = df[df.datadate.dt.year >= year_start]

    for col in df.columns[1:]:
        df[col][df[col] == ''] = None
        df[col] = df[col].astype(float)

    df['roiq'] = df['roiq'] / 100
    return df

def scrape_price_data(tic, year_start=2018):
    """
    Scrap closing price data for a firm.

    Parameters
    ----------
    tic : str
        Ticker of company for which data will be retrieved
    year_start : int
        Starting year for data records to retrieve

    Returns
    -------
    DataFrame
        Dataframe containing price data (identified by date)
    """

    t = Ticker(tic)

    year_string = str(year_start) + '-01-01'
    df = t.history(start=year_string)
    fields_to_keep = ['close']
    df = df[fields_to_keep]
    df = df.reset_index()
    df = df.rename(columns={'index': 'date', 'close': 'price'})

    return df

def add_dividends(qfirm, tic, set_div_type=False):
    """
    Adds annualized, split adjusted and error corrected dividends to each financial quarter of a firm.

    Parameters
    ----------
    qfirm : DataFrame
        Dataset containing quarterly financials of a company
    year_start : int
        Starting year for data records to retrieve

    Returns
    -------
    DataFrame
        Financial data containing annualized dividends
    """

    # Sleep (Avoid Yahoo Blockage)
    time.sleep(5)

    # ------ DIVIDEND DATA -------
    dividends = ms.get_dividends(tic, day_history=1200)

    # Unavailable data - continue
    if dividends.shape[0] == 0:
        # We can or cannot keep firms w/o dividends
        print('Continued (Err1)')
        qfirm['divq'] = 0
        return qfirm


    dividends['div_date'] = pd.to_datetime(dividends['div_date'], dayfirst=True)
    dividends['cyearq'] = dividends.div_date.dt.year

    # Remove large dividends (SDDs)
    dividends = dividends[dividends.divq < 5]

    # Check if firm pays quarterly, annual or bi-annual dividends
    aux = dividends[dividends.divq > 0].groupby(['cyearq'])['divq'].count()
    aux = aux[aux>=1]
    aux = aux.to_list()

    if not aux:
        print('Continued (Err2)')
        return df_merge
    elif set_div_type:
        counts = np.bincount(aux)
        div_type = np.argmax(counts)

        if div_type <= 2:
            dividends = dividends.groupby('cyearq').apply(lambda x: x.divq.nsmallest(2).sum())
        elif div_type > 2:
            dividends = dividends.groupby('cyearq').apply(lambda x: x.divq.nsmallest(4).sum())
    else:
        dividends = dividends.groupby('cyearq').apply(lambda x: x.divq.nsmallest(4).sum())

    dividends = dividends.to_frame(name='divy').reset_index()
    dividends['divy'] = round(dividends['divy'], 4)

    df_merge = pd.merge(qfirm, dividends, how='left', on=['cyearq'])
    df_merge['divy'].fillna(0, inplace=True)

    # Append to final quarterly dividend paying firm observations
    return df_merge


def getNewData(update=False, incomplete=False, year_start=2018):
    """
    Retrieve new financials for sample dividend paying firms, from Macrotrends and Yahoo Finance.

    Parameters
    ----------
    update : Boolean
        If True, appends retrieved records to already existing Macrotrends/Yahoo records.
        By default, gathers and saves data from Macrotrends/Yahoo as a new file.
    incomplete : Boolean
        Setting to true forces the function to resume a previously aborted or terminated scraping
        round, looking for a local file containing incomplete records.
        By default, starts new scraping round from scratch.

    year_start : int
        Starting year for data records to retrieve

    Returns
    -------
    DataFrame
        Dataset with all scraped financial fields (IS + BS + CFS + price + dividends)
    """

    # Get Compustat dataframe
    old_df = pickle.load(open('files/quarter_df.pickle', 'rb'))
    # Gather tickers
    tickers = old_df['tic'].unique()
    df_out = pd.DataFrame()

    tickers = [t for t in tickers if t not in ignore_list]

    # In order to update, there must be a pickle file available
    if update == True:
        print('Updating all the previously scraped data.')
        df_saved = pickle.load(open('files/scrap_df.pickle', 'rb'))
        for tic in tickers:
            # Sleep (Avoid Yahoo Blockage)
            time.sleep(3)

            firm_saved = df_saved[df_saved.tic == tic]

            df_is = scrape_quarter_income_statement(tic)
            df_bs = scrape_quarter_balance_sheet(tic)
            df_cfs = scrape_cf_yahoo(tic)
            df_ratios = scrape_quarter_ratios(tic)

            df_firm = pd.merge(df_is, df_bs, left_on='datadate', right_on='datadate', how='inner')
            df_firm = pd.merge(df_firm, df_ratios, left_on='datadate', right_on='datadate', how='inner')
            df_firm = pd.merge(df_firm, df_cfs, left_on='datadate', right_on='datadate', how='left')
            df_firm = df_firm.drop_duplicates(subset=['datadate']).sort_values(by=['datadate'])
            df_firm['tic'] = tic
            firm_out = firm_saved.append(df_firm)
            firm_out = firm_out.drop_duplicates(['datadate'])

            df_out = df_out.append(firm_out)

        pickle.dump(df_out, open('files/scrap_df.pickle', 'wb'))
        return df_out
    else:
        print('Reading all the data from 2018 onwards.')
        try:
            df_saved = pickle.load(open('files/scrap_df.pickle', 'rb'))
            print('Data already in disk.')
            df_out = df_saved
        except (OSError, IOError) as e:
            print('Data not in disk, reading from Yahoo/Macrotrends...')

            if incomplete == True:
                print('Previous scraping round incomplete... Will complete sample.')
                inc_df = pickle.load(open('files/scrap_inc_df.pickle', 'rb'))
                df_out = inc_df
                prev_t = inc_df['tic'].unique()
                ticker_list = [f for f in tickers if f not in prev_t]
            else:
                print('New scraping round.')
                ticker_list = tickers

            for tic in ticker_list:
                print('Loading ... ', tic)

                df_is = scrape_quarter_income_statement(tic)
                df_bs = scrape_quarter_balance_sheet(tic)
                df_cfs = scrape_cf_yahoo(tic)
                df_ratios = scrape_quarter_ratios(tic)


                df_firm = pd.merge(df_is, df_bs, left_on='datadate', right_on='datadate', how='inner')
                df_firm = pd.merge(df_firm, df_ratios, left_on='datadate', right_on='datadate', how='inner')
                df_firm = pd.merge(df_firm, df_cfs, left_on='datadate', right_on='datadate', how='left')
                df_firm = df_firm.drop_duplicates(subset=['datadate']).sort_values(by=['datadate'])
                #df_firm = pd.merge_asof(df_firm, df_prices, left_on='datadate', right_on='datadate')

                df_firm['tic'] = tic
                #df_firm = add_dividends(df_firm, tic)
                df_out = df_out.append(df_firm)

                pickle.dump(df_out, open('files/scrap_inc_df.pickle', 'wb'))

            pickle.dump(df_out, open('files/scrap_df.pickle', 'wb'))

        return df_out

def mergeUpdatedData(new_df):
    """
    Merge Compustat/CRSP records dataset, which includes processed dividends and financial ratios, with scraped fundamental
    data returned from getNewData(). The appended data will get the same processing as the Compustat/CRSP (2008 to 2018),
    resulting in a uniform dataset.
    :param new_df: Data to append to Compustat dataframe, generated by getNewData()
    :return: Compustat + Web scraped data dataframe, which contains financial quarter from 2008 to the present, with all
    relevant ratios
    """
    """
    Merge existing dataset, which includes processed dividends and financial ratios, with scraped fundamental
    data returned from getNewData(). After the data is merged, the missing financial ratios are added to the
    final dataset.

    Parameters
    ----------
    new_df : DataFrame
        Web scraped data, to be added to the main dataset

    Returns
    -------
    DataFrame
        Final updated dataset, containing all required fields and ratios
    """

    try:
        df_out = pickle.load(open('files/updated_quarter_df.pickle', 'rb'))
        return df_out
    except (OSError, IOError) as e:
        old_df = pickle.load(open('files/quarter_df.pickle', 'rb'))

        tickers = old_df['tic'].unique()
        upd_tickers = new_df['tic'].unique()

        n = 0
        df_out = pd.DataFrame()

        print('Merging COMPUSTAT data with updated data...')

        for tic in tickers:
            print(tic)

            # Data slice of firm
            old_firm = old_df[old_df.tic == tic]
            new_firm = new_df[new_df.tic == tic]

            # Old records
            last_record = old_firm[['datadate', 'fyearq', 'fqtr', 'rdq', 'Sector']].tail(4)

            # Get sector
            sector = last_record.iloc[0]['Sector']

            # Get last delay period between report date and release
            delay = (last_record.iloc[-1]['rdq'] - last_record.iloc[-1]['datadate'])#.dt.total_seconds()

            # Adjust financial quarter dates
            last_record['year'] = last_record['datadate'].dt.year
            last_record['month'] = last_record['datadate'].dt.month
            lr_values = last_record[['year', 'month', 'fyearq', 'fqtr']].reset_index(drop=True)

            fyear_map = dict()
            q_map = dict()
            for i in range(0, 4):
                fyear_map[int(lr_values.iloc[i, 3])] = float(lr_values.iloc[i, 2]) - float(lr_values.iloc[i, 0])
                q_map[int(lr_values.iloc[i, 1])] = float(lr_values.iloc[i, 3])

            if ((len(fyear_map) < 4) or (len(q_map) < 4)) or (2018 not in last_record['year'].tolist()) or (tic not in upd_tickers):
                updated_firm = rg.generateRatiosFromScrapedData(old_firm, new=False)
                updated_firm = rg.dropNonRatios(updated_firm)
                df_out = df_out.append(updated_firm)
                continue

            # Date labels
            new_firm['month'] = new_firm['datadate'].dt.month
            new_firm['year'] = new_firm['datadate'].dt.year
            old_firm['month'] = old_firm['datadate'].dt.month
            old_firm['year'] = old_firm['datadate'].dt.year

            new_firm['fqtr'] = new_firm['month'].map(q_map)
            new_firm['fyearq'] = new_firm['year'] + new_firm['fqtr'].map(fyear_map)

            # Add sector
            new_firm['Sector'] = sector

            # Compute financial report release date (estimative)
            new_firm['rdq'] = new_firm['datadate'] + delay
            new_firm['cyearq'] = new_firm.rdq.dt.year

            # Add dividends
            new_firm = add_dividends(new_firm, tic)

            # Correct prices for release date
            if 'price' in new_firm.columns:
                new_firm = new_firm.drop(['price'], axis=1)

            # Add closin price (based on release date of report)
            df_prices = ms.get_price(tic, year_start=2018)
            df_prices['date'] = pd.to_datetime(df_prices['date'])
            
            new_firm_with_price = pd.merge_asof(new_firm, df_prices, left_on='rdq', right_on='date')

            # Final merging operation
            updated_firm = old_firm.append(new_firm_with_price)
            updated_firm = updated_firm.drop_duplicates(subset=['year', 'month'])
            updated_firm = updated_firm.drop(columns=['year', 'month'])
            updated_firm = rg.generateRatiosFromScrapedData(updated_firm)
            updated_firm = rg.dropNonRatios(updated_firm)

            df_out = df_out.append(updated_firm)

        df_out = df_out.reset_index(drop=True)

        # Save final data
        print('Saving... Done.')
        pickle.dump(df_out, open('files/updated_quarter_df.pickle', 'wb'))

        return df_out
