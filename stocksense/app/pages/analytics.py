import datetime as dt
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from database_handler import DatabaseHandler
from plotly.subplots import make_subplots

pd.options.mode.chained_assignment = None  # default='warn'

MILLION = 1000000
MARGIN = {"l": 0, "r": 10, "b": 10, "t": 25}


def list_stocks():
    db = DatabaseHandler()
    stocks = db.fetch_stock().to_pandas()
    return sorted(
        stocks.loc[
            stocks.spx_status == 1,  # noqa: E712
            "tic",
        ].values.tolist()
    )


def date_breaks(df, date_col="date"):
    dt_all = pd.date_range(start=df[date_col].min(), end=df[date_col].max())
    dt_obs = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(df[date_col])]
    return [d for d in dt_all.strftime("%Y-%m-%d").tolist() if d not in dt_obs]


@st.cache_data(show_spinner="Fetching index data...", max_entries=10)
def load_index_data():
    db = DatabaseHandler()
    return db.fetch_index_data().to_pandas()


@st.cache_data(show_spinner="Fetching processed data...", max_entries=1)
def load_processed_data():
    """
    Read most recently processed dataset.
    """

    directory_path = Path("data/1_work_data/processed")
    csv_files = directory_path.glob("*.csv")

    date_files = [
        (file, dt.datetime.strptime(file.stem.split("_")[-1], "%Y-%m-%d")) for file in csv_files
    ]
    if date_files:
        most_recent_file = max(date_files, key=lambda x: x[1])[0]
        return pl.read_csv(most_recent_file, try_parse_dates=True).to_pandas()
    else:
        raise FileNotFoundError


@st.cache_data(show_spinner="Fetching stock data...", max_entries=10)
def load_stock_data(ticker):
    db = DatabaseHandler()
    stock = db.fetch_stock(ticker).to_pandas()
    info = db.fetch_info(ticker).to_pandas()
    market = db.fetch_market_data(ticker).to_pandas()
    financials = db.fetch_financial_data(ticker).to_pandas()
    insider = db.fetch_insider_data(ticker).to_pandas()
    return stock, info, market, financials, insider


def display_stock_info(stock, info):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("**General Info**")
        st.markdown(f"**Sector**: {stock.loc[0, 'sector']}")
        st.markdown(f"**Last price**: {(info.loc[0, 'curr_price']):.2f} $")
        st.markdown(f"**Market Cap**: {(info.loc[0, 'market_cap'] / MILLION):.2f} M$")
        st.markdown(f"**Out. Shares**: {(info.loc[0, 'shares_outstanding'] / MILLION):.2f} M")
        st.markdown(f"**Volume**: {(info.loc[0, 'volume'])} M$")
        st.markdown(f"**Beta**: {(info.loc[0, 'beta']):.3f}")
        st.markdown(
            "**Enterprise Value**: " f"{(info.loc[0, 'enterprise_value'] / MILLION):.2f} M$"
        )
        st.divider()
        st.markdown(f"**Trailing PE**: {(info.loc[0, 'fiftytwo_wc']):.2f}")
        st.markdown(f"**Forward PE**: {(info.loc[0, 'short_ratio']):.2f}")
        st.markdown(f"**Forward EPS**: {(info.loc[0, 'forward_eps']):.2f}")
        st.markdown(f"**Price-to-Book**: {(info.loc[0, 'price_book']):.2f}")
    with col2:
        st.subheader("**Analyst Information**")
        st.markdown(f"**Risk**: {(info.loc[0, 'risk'])}")
        st.markdown(f"**Target Low**: {(info.loc[0, 'target_low']):.2f} $")
        st.markdown(f"**Target Mean**: {(info.loc[0, 'target_mean']):.2f} $")
        st.markdown(f"**Target High**: {(info.loc[0, 'target_high']):.2f} $")
        st.markdown(f"**Recommendation**: {(info.loc[0, 'rec_key'])}")


def plot_market_data(df, index_df):
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
        specs=[[{"secondary_y": True}], [{}]],
    )

    fig.update_xaxes(rangebreaks=[{"values": date_breaks(df)}])
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["adj_close" if adj_close else "close"],
            name="Stock Price",
            marker_color="orangered",
            mode="lines",
        ),
        row=1,
        col=1,
    )
    if show_sp:
        fig.add_trace(
            go.Scatter(
                x=index_df["date"],
                y=index_df["adj_close" if adj_close else "close"],
                name="S&P500 Price",
                marker_color="green",
                mode="lines",
            ),
            secondary_y=True,
            row=1,
            col=1,
        )

    colors = ["#27AE60" if dif >= 0 else "#B03A2E" for dif in df["close"].diff().values.tolist()]

    fig.add_trace(
        go.Bar(x=df["date"], y=df["volume"], showlegend=False, marker_color=colors),
        row=2,
        col=1,
    )
    title = (
        "Daily Adjusted Close Price and Volume Data"
        if adj_close
        else "Daily Close Price and Volume Data"
    )
    layout = go.Layout(title=title, height=500, margin=MARGIN)
    fig.update_layout(layout, template="plotly_dark")
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True, theme=None)


def plot_financial_data(df):
    """
    Plots financials bar charts.
    """
    col = st.selectbox("Select", df.columns[3:], key="financial")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["rdq"], y=df[col], name=f"{col}", marker_color="orangered"))
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True, theme=None)


def plot_financial_analysis(df):
    """
    Creates a comprehensive financial dashboard with multiple visualization types:
    - Main metrics trends (Revenue, Net Income, EPS)
    - Growth rates
    - Margins analysis
    - Custom metric selector
    """

    growth_alias = ["qoq", "yoy", "2y", "return"]
    growth_vars = [f for f in df.columns if any(xf in f for xf in growth_alias)]
    for col in growth_vars:
        if col in df.columns:
            df[col] = df[col] * 100

    ratio_vars = list(df.columns[15:])
    margins = st.multiselect(
        "Select metric",
        ratio_vars,
        ["roa", "gpm", "dr"],
    )
    fig = go.Figure()
    for margin in margins:
        fig.add_trace(
            go.Scatter(
                x=df["rdq"],
                y=df[margin],
                name=margin.replace("_", " ").title(),
                mode="lines+markers",
            )
        )
    fig.update_layout(
        height=400,
        template="plotly_dark",
        title_text="Financial Metric Overview",
        margin={"l": 10, "r": 10, "b": 10, "t": 30},
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)


def plot_insider_data(df):
    """
    Plots scatter plot for insider trading data.
    """

    df["value"] = df["value"].replace({r"\$": "", ",": ""}, regex=True).astype(float).abs()

    fig = px.scatter(
        df,
        x="filling_date",
        y="value",
        hover_name="owner_name",
        hover_data=list(df.columns),
        title="Insider Trading Data",
        color="transaction_type",
        labels={"filling_date": "Filling Date", "last_price": "Last stock price"},
    )
    fig.update_layout(template="plotly_dark")
    fig.update_traces(marker={"size": 10})
    st.plotly_chart(fig, use_container_width=True, theme=None)


def plot_processed_data(df):
    """
    Plots processed feature set bar charts.
    """
    col = st.selectbox("Select", df.columns[15:], key="proc")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["tdq"], y=df[col], name=f"{col}", marker_color="orangered"))
    st.plotly_chart(fig, use_container_width=True)


def main():
    """
    Main app script.
    """

    st.set_page_config(layout="wide", page_title="Stock Data Analytics", page_icon="📈")
    st.sidebar.title("Stocksense App")
    st.sidebar.success("Select page")

    st.sidebar.page_link("home.py", label="Home", icon="🏠")
    st.sidebar.page_link("pages/overview.py", label="Market Overview", icon="🌎")
    st.sidebar.page_link("pages/analytics.py", label="Stock Analytics", icon="📈")
    st.sidebar.page_link("pages/insights.py", label="Stock Picks", icon="🔮")
    st.sidebar.divider()
    st.sidebar.header("Options: ")

    ticker = st.sidebar.selectbox(
        "Pick stock", options=list_stocks(), help="Select a ticker", key="ticker"
    )

    selected_range = st.sidebar.select_slider(
        "Select period",
        options=["1M", "6M", "YTD", "1Y", "2Y", "5Y", "All"],
        value="2Y",
    )

    stock, info, market, financials, insider = load_stock_data(ticker)
    processed = load_processed_data()
    sp = load_index_data()

    name = stock.loc[0, :].values.flatten().tolist()[1]
    min_date = market["date"].min()
    max_date = market["date"].max()

    start_dates = {
        "1M": max_date + pd.DateOffset(months=-1),
        "6M": max_date + pd.DateOffset(months=-6),
        "YTD": max_date.replace(month=1, day=1),
        "1Y": max_date + pd.DateOffset(months=-12),
        "2Y": max_date + pd.DateOffset(months=-24),
        "5Y": max_date + pd.DateOffset(months=-60),
        "All": min_date,
    }

    st.session_state.tic = ticker
    st.session_state.page_subheader = f"{name} ({ticker})"

    st.subheader(st.session_state.page_subheader)
    st.markdown(f"**Last update**: {stock.loc[0, 'last_update']}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Status", "Market", "Financials", "Insider Trading", "Feature Analysis"]
    )
    with tab1:
        display_stock_info(stock, info)
    with tab2:
        mdf = market[(market["date"] >= start_dates[selected_range]) & (market["date"] <= max_date)]
        idf = sp[(sp["date"] >= start_dates[selected_range]) & (sp["date"] <= max_date)]
        plot_market_data(mdf, idf)
    with tab3:
        fdf = financials.loc[
            (financials["rdq"] >= start_dates[selected_range]) & (financials["rdq"] <= max_date)
        ]
        plot_financial_data(fdf)
    with tab4:
        indf = insider.loc[
            (insider["filling_date"] >= start_dates[selected_range])
            & (insider["filling_date"] <= max_date)
        ]
        plot_insider_data(indf)
    with tab5:
        pdf = processed.loc[
            (processed["tic"] == ticker)
            & (processed["tdq"] >= start_dates[selected_range])
            & (processed["tdq"] <= max_date)
        ]
        plot_financial_analysis(pdf)


if __name__ == "__main__":
    main()
