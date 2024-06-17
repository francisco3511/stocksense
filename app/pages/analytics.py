
import streamlit as st
import pandas as pd

# plotting
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from st_aggrid import GridOptionsBuilder, AgGrid, ColumnsAutoSizeMode
from st_aggrid.shared import JsCode

import talib as ta
import yfinance as yf

from pipeline import Stock
from utils import get_stock_metadata

MARGIN = dict(l=0,r=10,b=10,t=25)

# JsCode to highlight cells when surprise < 0
cellsytle_jscode = JsCode(
    """
        function(params) {
            if (params.data.Surprise < 0) {
                return {
                    'color': 'white',
                    'backgroundColor': '#A93226'
                }
            }
        };
    """
)

st.set_page_config(layout='wide', page_title='StockSense', page_icon='📈')


def list_stocks():
    
    # read control file
    control_df = pd.read_csv(
        './data/1_work_data/SP500.csv',
        sep=';'
    )

    # return SP500 constituents
    return control_df.loc[
        control_df['Status'] == True,  # noqa: E712
        'Symbol'
    ].values.tolist()


def date_breaks(df):
    # build complete timeline from start date to end date
    dt_all = pd.date_range(start=df.iloc[0]['Date'], end=df.iloc[-1]['Date'])
    # retrieve the dates that are in the original datset
    dt_obs = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(df['Date'])]
    # define dates with missing values
    return [d for d in dt_all.strftime("%Y-%m-%d").tolist() if not d in dt_obs]


def categorize_number(number):
    if number < 0:
        return "Negative"
    elif number == 0:
        return "Expected"
    else:
        return "Positive"


@st.cache_data(show_spinner="Fetching stock data...", max_entries=10)
def load_stock_data(ticker, update=False):
    
    # get stock data handler
    stock = Stock(ticker)
    
    # get stock info
    info = get_stock_metadata(ticker)
    
    if update:
        # update stock data
        stock.update_stock_data(method='yh')
        
    return stock, info


def format_number(number):
    """
    Formats a number to display in Trillion(T), Billion(B) or Million(M)
    """
    if abs(number) >= 1_000_000_000_000:
        return f"{number / 1_000_000_000_000:.2f}T"
    elif abs(number) >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f}B"
    elif abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"
    else:
        return str(number)
    

def plot_bubble_chart(data):
    if data.empty:
        print("No data available to plot.")
        return
    
    fig = px.scatter(
        data,
        x='Date',
        y='Average Price',
        size='Shares Traded',
        hover_name='Insider Name',
        hover_data=['Position', 'Total Amount', 'Shares Held'],
        title='Insider Trading Data (Bubble Chart)',
        labels={'Date': 'Transaction Date', 'Average Price': 'Average Price per Share'}
    )

    fig.update_traces(marker=dict(line=dict(width=1, color='Black')))
    fig.show()


st.sidebar.header("Options: ")

ticker = st.sidebar.selectbox(
    'Choose Ticker',
    options=list_stocks(),
    help = 'Select a ticker',
    key='ticker'
)

selected_range = st.sidebar.select_slider(
    'Select period',
    options=['1M', '6M', 'YTD', '1Y', 'All'],
    value='1Y'
)

update = st.sidebar.button("Fetch updates")

# retrieve market data
stock, info = load_stock_data(ticker, update)

# get market and financial data
market_df = stock.get_market_data()

# Calculate the MAs for graphs
market_df['SMA-50'] = ta.SMA(market_df['Close'],timeperiod=50)
market_df['SMA-200'] = ta.SMA(market_df['Close'], timeperiod=200)
#market_df['RSI'] = ta.RSI(market_df['Close'], timeperiod=14)

fin_df = stock.get_financial_data()

fin_df['Category'] = fin_df['surprise_pct'].apply(categorize_number)

min_date = market_df.iloc[0]['Date']
max_date = market_df.iloc[-1]['Date']

start_m1 = max_date + pd.DateOffset(months=-1)
start_m6 = max_date + pd.DateOffset(months=-6)
start_ytd = max_date.replace(month=1, day=1)
start_y1 = max_date + pd.DateOffset(months=-12)
start_all = min_date
end_date = max_date

match selected_range:
    case '1M':
        start_date = start_m1
    case '6M':
        start_date = start_m6
    case 'YTD':
        start_date = start_ytd
    case '1Y' :
        start_date = start_y1
    case default:
        start_date = start_all

# Subheader with company name and symbol
st.session_state.page_subheader = f'{stock.name} ({stock.tic})'
st.subheader(st.session_state.page_subheader)
st.divider()

tab1, tab2, tab3 = st.tabs(["Market", "Financials", "Stocksensing"])

# Market tab
with tab1:
    
    # Filter dates
    mdf = market_df[(market_df['Date'] >= start_date) & (market_df['Date'] <= end_date)]

    col1, col2, col3, col4 = st.columns([1,1,1,4.5])
    with col1:
        st.text('Sector')
        st.text(info['industry'])
        st.divider()
        st.text('Market Cap')
        st.text(format_number(info['marketCap']))
        st.divider()
        st.text('Previous Close')
        st.text(info['previousClose'])
        st.divider()
        st.text('Beta')
        st.text('{:.2f}'.format(info['beta']))
    with col2:
        st.text('Average Volume')
        st.text('{:,}'.format(info['averageVolume']))
        st.divider()
        st.text('Fwd Div & Yield')
        if 'dividendRate' in info:
            st.text('{0} ({1:.2f})%'.format(info['dividendRate'], info['dividendYield'] * 100))
        else:
            st.text('NA')
        st.divider()
    
    with col3:
        st.text('EPS (TTM)')
        st.text(info['trailingEps'])
        st.divider()
        st.text('PEG Ratio (TTM)')
        if 'trailingPegRatio' in info:
            st.text(f'{0:.2f}'.format(info['trailingPegRatio']))
        else:
            st.text('NA')
        st.divider()
        st.text('PE Ratio (TTM)')
        if 'trailingPE' in info:
            st.text('{0:.2f}'.format(info['trailingPE']))
        else:
            st.text('NA')
        st.divider()
        st.text('Forward PE Ratio')
        if 'forwardPE' in info:
            st.text('{0:.2f}'.format(info['forwardPE']))
        else:
            st.text('NA')

    with col4:
        
        # Construct a 2 x 1 Plotly figure 
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.01, shared_xaxes=True)
        
        # Remove dates without values
        fig.update_xaxes(rangebreaks=[dict(values=date_breaks(mdf))])

        # Plot the Price chart
        fig.add_trace(
            go.Scatter(
                x=mdf['Date'],
                y=mdf['Adj Close'],
                name='Price',
                marker_color='orangered',
                mode='lines'
            ),
            row=1,
            col=1
        )
        
        # Color maps for different MAs
        COLORS_MAPPER = {
            'SMA-50': '#38BEC9',
            'SMA-200': '#E67E22',
        }
        
        for ma, col in COLORS_MAPPER.items():
            fig.add_trace(go.Scatter(x=mdf['Date'], y=mdf[ma], name=ma, marker_color=col))
            
        # colors for the Bar chart
        colors = ['#27AE60' if dif >= 0 else '#B03A2E' for dif in mdf['Close'].diff().values.tolist()]
        
        # Adds the volume as a bar chart
        fig.add_trace(go.Bar(x=mdf['Date'], y=mdf['Volume'], showlegend=False, marker_color=colors), row=2, col=1)
        
        # add title
        layout = go.Layout(title='Price, MA and Volume', height=500, margin=MARGIN)
        
        fig.update_layout(layout)
        st.plotly_chart(fig, theme='streamlit', use_container_width=True)
        
with tab2:
    
    # filter dates
    fdf = fin_df[(fin_df['datadate'] >= start_date) & (fin_df['datadate'] <= end_date)]

    # set up tabs
    earn_tab, eps_tab = st.tabs(["Earnings", "EPS"])
    
    with eps_tab:
           
        gb = GridOptionsBuilder()
        # Pagination is set to 10 rows per page
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)
        # Apply the jscode to display the row in red for negative EPS
        gb.configure_default_column(cellStyle=cellsytle_jscode)

        gb.configure_column(
            field="datadate",
            valueFormatter="value != undefined ? new Date(value).toLocaleString('en-AU', {dateStyle:'medium'}): ''",
            width=125
        )
        gb.configure_column(
            field="rdq",
            valueFormatter="value != undefined ? new Date(value).toLocaleString('en-AU', {dateStyle:'medium'}): ''",
            width=125
        )
        gb.configure_column(
            field="eps_rep",
            type=["numericColumn"],
            width=120
        )
        gb.configure_column(
            field="eps_est",
            type=["numericColumn"],
            width=125
        )

        gb.configure_column(
            field="surprise_pct",
            valueFormatter="value + '%'",
            width=147
        )
        gridOptions = gb.build()

        col1, col2 = st.columns([3, 1])
        with col1:
            AgGrid(
                fdf[::-1], 
                gridOptions=gridOptions, theme="balham", 
                #columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
                allow_unsafe_jscode=True)
        with col2:
            st.write("")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fdf["rdq"], y=fdf["eps_rep"],
            name="Reported EPS",
            marker_color="#A93226"
        ))

        fig.add_trace(go.Scatter(
            x=fdf["rdq"], y=fdf["eps_est"],
            name="Estimated EPS",
            marker_color="#F5B7B1"
        ))

        # Set options common to all traces with fig.update_traces
        fig.update_traces(mode="markers", marker_line_width=2, marker_size=10)
        fig.update_layout(title="Reported/Estimated EPS", margin=MARGIN, legend=dict(orientation="h"))
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

        # Normlaize suprise column as it contains negative values
        # norm = ((df['Surprise'] - df['Surprise'].min()) / (df['Surprise'].max() - df['Surprise'].min())) * 100

        fig = px.scatter(fdf, x="rdq", y="eps_rep", color="Category",
                        color_discrete_map = {'Negative': "#A93226", 'Expected': "#7FB3D5", 'Positive': "#1E8449"})
        fig.update_traces(marker_size=10)
        fig.update_layout(title="Reported EPS (Categorized)", xaxis_title=None, legend_title=None, yaxis_title=None,
                        margin=MARGIN, legend=dict(orientation="h"))
        st.plotly_chart(fig, theme='streamlit', use_container_width=True)
        
with tab3:
    
    