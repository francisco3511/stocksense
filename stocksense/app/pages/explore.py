
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from database_handler import DatabaseHandler

MARGIN = dict(l=0, r=10, b=10, t=25)


def list_stocks():

    # read db control table
    db = DatabaseHandler()
    stocks = db.fetch_stock().to_pandas()

    # return SP500 constituents
    return sorted(
        stocks.loc[
            stocks.spx_status == 1,  # noqa: E712
            'tic'
        ].values.tolist()
    )


def categorize_number(number):
    if number < 0:
        return "Negative"
    elif number == 0:
        return "Expected"
    else:
        return "Positive"


def date_breaks(df, date_col='date'):
    dt_all = pd.date_range(start=df[date_col].min(), end=df[date_col].max())
    dt_obs = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(df[date_col])]
    return [d for d in dt_all.strftime("%Y-%m-%d").tolist() if d not in dt_obs]


@st.cache_data(show_spinner="Fetching index data...", max_entries=10)
def load_index_data():
    db = DatabaseHandler()
    return db.fetch_index_data().to_pandas()


@st.cache_data(show_spinner="Fetching stock data...", max_entries=10)
def load_stock_data(ticker):
    db = DatabaseHandler()
    stock = db.fetch_stock(ticker).to_pandas()
    market = db.fetch_market_data(ticker).to_pandas()
    financials = db.fetch_financial_data(ticker).to_pandas()
    insider = db.fetch_insider_data(ticker).to_pandas()
    return stock, market, financials, insider


def plot_market_data(df, index_df, start_dates, max_date):
    """
    Plots market data (price and volume).
    Adds options to
    """

    col1, col2 = st.columns(2)
    with col1:
        adj_close = st.checkbox("Adjusted Close")
    with col2:
        show_sp = st.checkbox("S&P500")

    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.01,
        shared_xaxes=True,
        specs=[[{"secondary_y": True}], [{}]]
    )

    fig.update_xaxes(rangebreaks=[dict(values=date_breaks(df))])
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['adj_close' if adj_close else 'close'],
            name='Price',
            marker_color='orangered',
            mode='lines'
        ),
        row=1,
        col=1
    )

    if show_sp:
        idf = sp[(sp['date'] >= start_dates[selected_range]) & (sp['date'] <= max_date)]
        fig.add_trace(
            go.Scatter(
                x=idf['date'],
                y=idf['adj_close' if adj_close else 'close'],
                name='S&P500 Price',
                marker_color='blue',
                mode='lines'
            ),
            secondary_y=True,
            row=1,
            col=1
        )

    colors = [
        '#27AE60' if dif >= 0 else '#B03A2E'
        for dif in mdf['close'].diff().values.tolist()
    ]

    fig.add_trace(
        go.Bar(x=mdf['date'], y=mdf['volume'], showlegend=False, marker_color=colors),
        row=2,
        col=1
    )

    title = (
        "Daily Adjusted Close Price and Volume Data" if adj_close
        else "Daily Close Price and Volume Data"
    )
    layout = go.Layout(title=title, height=500, margin=MARGIN)
    fig.update_layout(layout, template='ggplot2')
    st.plotly_chart(fig, use_container_width=True)


def plot_insider_data(df):
    """
    Plots scatter plot for insider trading data.
    """

    df['qty'] = df['qty'].replace({r"\$": "", ",": ""}, regex=True).astype(float)
    df['shares_held'] = df['shares_held'].replace({r"\$": "", ",": ""}, regex=True).astype(float)
    df['value'] = df['value'].replace({r"\$": "", ",": ""}, regex=True).astype(float).abs()
    df['last_price'] = df['last_price'].replace({r"\$": "", ",": ""}, regex=True).astype(float)

    fig = px.scatter(
        df,
        x='filling_date',
        y='value',
        hover_name='owner_name',
        hover_data=list(df.columns),
        title='Insider Trading Data (Bubble Chart)',
        color='transaction_type',
        labels={'filling_date': 'Filling Date', 'last_price': 'Last stock price'}
    )
    fig.update_layout(
        template='plotly'
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)


def main():
    """
    Main app script.
    """

    # Set base configuration
    st.set_page_config(
        layout="wide",
        page_title="Explore Stock Data",
        page_icon="🌎"
    )

    st.sidebar.title("Stocksense App")
    st.sidebar.success("Select page")

    # Add pages links
    st.sidebar.page_link("home.py", label="Home", icon="🏠")
    st.sidebar.page_link("pages/explore.py", label="Explore Stock Data", icon="🌎")
    st.sidebar.page_link("pages/analytics.py", label="Stock Analytics", icon="📈")
    st.sidebar.page_link("pages/insights.py", label="Stock Picks", icon="🔮")
    st.sidebar.divider()

    st.sidebar.header("Options: ")

    ticker = st.sidebar.selectbox(
        "Pick stock",
        options=list_stocks(),
        help="Select a ticker",
        key="ticker"
    )

    selected_range = st.sidebar.select_slider(
        'Select period',
        options=['1M', '6M', 'YTD', '1Y', '2Y', '5Y', 'All'],
        value='2Y'
    )

    # retrieve market data
    stock, market, financials, insider = load_stock_data(ticker)
    sp = load_index_data()

    name = stock.loc[0, :].values.flatten().tolist()[1]
    sector = stock.loc[0, :].values.flatten().tolist()[2]
    min_date = market['date'].min()
    max_date = market['date'].max()
    financials['Category'] = financials['surprise_pct'].apply(categorize_number)

    start_dates = {
        '1M': max_date + pd.DateOffset(months=-1),
        '6M': max_date + pd.DateOffset(months=-6),
        'YTD': max_date.replace(month=1, day=1),
        '1Y': max_date + pd.DateOffset(months=-12),
        '2Y': max_date + pd.DateOffset(months=-24),
        '5Y': max_date + pd.DateOffset(months=-60),
        'All': min_date
    }

    # subheader with company name and symbol
    st.session_state.page_subheader = f'{name} ({ticker})'
    st.subheader(st.session_state.page_subheader)
    st.markdown(f"**Sector**: {sector}")
    st.markdown(f"**Last updated**: {stock.loc[0, 'last_update']}")

    tab1, tab2, tab3 = st.tabs(["Market", "Financials", "Insider Trading"])

    with tab1:
        # filter data
        mdf = market[(market['date'] >= start_dates[selected_range]) & (market['date'] <= max_date)]
        # options
        col1, col2 = st.columns(2)
        with col1:
            adj_close = st.checkbox("Adjusted Close")
        with col2:
            show_sp = st.checkbox("S&P500")

        fig = make_subplots(
            rows=2,
            cols=1,
            vertical_spacing=0.01,
            shared_xaxes=True,
            specs=[[{"secondary_y": True}], [{}]]
        )

        fig.update_xaxes(rangebreaks=[dict(values=date_breaks(mdf))])
        fig.add_trace(
            go.Scatter(
                x=mdf['date'],
                y=mdf['adj_close' if adj_close else 'close'],
                name='Price',
                marker_color='orangered',
                mode='lines'
            ),
            row=1,
            col=1
        )

        if show_sp:
            idf = sp[(sp['date'] >= start_dates[selected_range]) & (sp['date'] <= max_date)]
            fig.add_trace(
                go.Scatter(
                    x=idf['date'],
                    y=idf['adj_close' if adj_close else 'close'],
                    name='S&P500 Price',
                    marker_color='blue',
                    mode='lines'
                ),
                secondary_y=True,
                row=1,
                col=1
            )

        colors = [
            '#27AE60' if dif >= 0 else '#B03A2E'
            for dif in mdf['close'].diff().values.tolist()
        ]

        fig.add_trace(
            go.Bar(x=mdf['date'], y=mdf['volume'], showlegend=False, marker_color=colors),
            row=2,
            col=1
        )

        title = (
            "Daily Adjusted Close Price and Volume Data" if adj_close
            else "Daily Close Price and Volume Data"
        )
        layout = go.Layout(title=title, height=500, margin=MARGIN)
        fig.update_layout(layout, template='ggplot2')
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        # filter data
        fdf = financials[
            (financials['rdq'] >= start_dates[selected_range]) &
            (financials['rdq'] <= max_date)
        ]

        fin_col = st.selectbox('Select', financials.columns[3:], key='financial')
        fig_fin = go.Figure()
        fig_fin.add_trace(go.Bar(
            x=fdf['rdq'],
            y=fdf[fin_col],
            name=f"{fin_col}",
            marker_color='orangered'
        ))
        st.plotly_chart(fig_fin, use_container_width=True)
    with tab3:
        # filter data
        indf = insider[
            (insider['filling_date'] >= start_dates[selected_range]) &
            (insider['filling_date'] <= max_date)
        ]
        plot_bubble_chart(indf)


if __name__ == "__main__":
    main()
