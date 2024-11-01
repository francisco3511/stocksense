
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from database_handler import DatabaseHandler

pd.set_option('future.no_silent_downcasting', True)


@st.cache_data(show_spinner="Fetching data...", max_entries=3)
def load_sp500_data():
    """
    Loads the S&P 500 data from the database.

    Returns
    -------
    pd.DataFrame
        Processed S&P 500 data.
    """
    db = DatabaseHandler()
    stocks = db.fetch_stock().to_pandas()
    stocks = stocks.loc[stocks.spx_status == 1]

    info = db.fetch_info().to_pandas()
    stock_df = stocks.merge(info, how='left', on='tic')

    financials = db.fetch_financial_data().to_pandas()
    financials['rdq'] = pd.to_datetime(financials['rdq'])
    financials = (
        financials
        .sort_values('rdq', ascending=False)
        .groupby('tic')
        .first()
        .reset_index()
    )
    stock_df = stock_df.merge(financials, how='left', on='tic')
    return stock_df


def plot_sector_distribution(data):
    """
    Plots the sector distribution of the S&P 500.

    Parameters
    ----------
    data : pd.DataFrame
        Processed S&P 500 data.
    """
    sector_counts = data['sector'].value_counts()
    fig = px.pie(
        values=sector_counts.values,
        names=sector_counts.index,
        title="S&P 500 Sector Distribution",
        template='plotly_dark'
    )
    st.plotly_chart(fig, use_container_width=True)


def show_recent_earnings(data):
    """
    Shows the recent earnings of the S&P 500.

    Parameters
    ----------
    data : pd.DataFrame
        Processed S&P 500 data.
    """
    df = data.sort_values('rdq', ascending=False).head(10)
    df = df[['tic', 'rdq', 'sector', 'curr_price', 'saleq', 'surprise_pct']]
    st.dataframe(
        df,
        column_config={
            "tic": "Stock",
            "rdq": st.column_config.DateColumn(
                "Earnings Date",
                format="YYYY-MM-DD"
            ),
            "sector": "Sector",
            "curr_price": st.column_config.NumberColumn(
                "Current Price",
                format="$%.2f"
            ),
            "saleq": st.column_config.NumberColumn(
                "Sales",
                format="$%.2f"
            ),
            "surprise_pct": st.column_config.NumberColumn(
                "Surprise %",
                format="$%.2f"
            ),
        },
        hide_index=True
    )


def show_market_summary(data):
    """
    Shows the market summary of the S&P 500.

    Parameters
    ----------
    data : pd.DataFrame
        Processed S&P 500 data.
    """

    data['trailing_pe'] = data['trailing_pe'].astype(float)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    total_market_cap = data['market_cap'].sum() / 1e12
    avg_pe = data['trailing_pe'].mean(skipna=True)
    avg_target_upside = (
        (data['target_mean'] - data['curr_price']) / data['curr_price']
    ).mean() * 100

    summary = {
        "Total Companies": len(data),
        "Total Market Cap": f"${total_market_cap:.2f}T",
        "Average P/E": f"{avg_pe:.2f}",
        "Avg Target Upside": f"{avg_target_upside:.1f}%"
    }

    st.subheader("Market Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Companies", summary["Total Companies"])
    with col2:
        st.metric("Total Market Cap", summary["Total Market Cap"])
    with col3:
        st.metric("Average P/E", summary["Average P/E"])
    with col4:
        st.metric("Avg Target Upside", summary["Avg Target Upside"])


def main():
    """
    Main app script.
    """

    st.set_page_config(
        layout="wide",
        page_title="Market Overview",
        page_icon="🌎"
    )
    st.sidebar.title("Stocksense App")
    st.sidebar.success("Select page")

    st.sidebar.page_link("home.py", label="Home", icon="🏠")
    st.sidebar.page_link("pages/overview.py", label="Market Overview", icon="🌎")
    st.sidebar.page_link("pages/analytics.py", label="Stock Analytics", icon="📈")
    st.sidebar.page_link("pages/insights.py", label="Stock Picks", icon="🔮")
    st.sidebar.divider()

    data = load_sp500_data()

    st.title("S&P 500 Market Overview")
    st.divider()

    show_market_summary(data)
    st.divider()

    col1, col2 = st.columns([2, 2])
    with col1:
        plot_sector_distribution(data)
    with col2:
        show_recent_earnings(data)


if __name__ == "__main__":
    main()
