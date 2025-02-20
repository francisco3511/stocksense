import datetime as dt

import pandas as pd
import plotly.express as px
import streamlit as st

from stocksense.config import PORTFOLIO_DIR
from stocksense.database import DatabaseHandler


@st.cache_data(show_spinner="Loading stock data...", max_entries=10)
def load_stock_data():
    db = DatabaseHandler()
    return db.fetch_stock().to_pandas()


def get_available_portfolios():
    """
    Get all available portfolio files.
    """
    portfolio_files = list(PORTFOLIO_DIR.glob("portfolio_*.xlsx"))
    dates = [dt.datetime.strptime(f.stem.split("_")[1], "%Y-%m-%d").date() for f in portfolio_files]
    return sorted(dates, reverse=True)


def load_portfolio(trade_date):
    """
    Load portfolio for a specific trade date.
    """
    portfolio_file = PORTFOLIO_DIR / f"portfolio_{trade_date}.xlsx"
    if not portfolio_file.exists():
        st.error(f"No portfolio found for trade date {trade_date}")
        return None
    return pd.read_excel(portfolio_file)


def plot_sector_distribution(portfolio_data):
    """
    Plot sector distribution of selected stocks.
    """
    sector_dist = portfolio_data.groupby("Sector")["Weight"].sum().reset_index()
    fig = px.pie(
        sector_dist,
        values="Weight",
        names="Sector",
        title="Sector Distribution",
        template="plotly_dark",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def plot_weight_distribution(portfolio_data):
    """
    Plot weight distribution of top 10 stocks.
    """
    top_10 = portfolio_data.nlargest(10, "Weight")
    fig = px.pie(
        top_10,
        values="Weight",
        names="Ticker",
        title="Top 10 Holdings by Weight",
        template="plotly_dark",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def display_portfolio_metrics(portfolio_data):
    """
    Display key portfolio metrics.
    """
    # Convert percentage strings to floats for calculations
    portfolio_data['Weight'] = portfolio_data['Weight'].str.rstrip('%').astype(float) / 100

    avg_model_score = portfolio_data['Model Score'].mean()
    num_sectors = portfolio_data['Sector'].nunique()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average Model Score", f"{avg_model_score:.1f}")
    with col2:
        st.metric("Sectors", num_sectors)


def main():
    """Insights main script."""
    st.set_page_config(layout="wide", page_title="Portfolio Insights", page_icon="💼")

    # Sidebar navigation
    st.sidebar.title("Stocksense App")
    st.sidebar.success("Select page")
    st.sidebar.page_link("home.py", label="Home", icon="🏠")
    st.sidebar.page_link("pages/overview.py", label="Market Overview", icon="🌎")
    st.sidebar.page_link("pages/analytics.py", label="Stock Analytics", icon="📈")
    st.sidebar.page_link("pages/insights.py", label="Portfolio Insights", icon="💼")
    st.sidebar.divider()

    # Main content
    st.title("Portfolio Insights 💼")
    st.divider()

    # Portfolio selection
    available_dates = get_available_portfolios()
    if not available_dates:
        st.warning("No portfolio files found.")
        return

    trade_date = st.selectbox(
        "Select Portfolio Date",
        available_dates,
        format_func=lambda x: x.strftime("%B %d, %Y")
    )

    portfolio = load_portfolio(trade_date)

    if portfolio is not None:
        st.divider()
        display_portfolio_metrics(portfolio)
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            weight_fig = plot_weight_distribution(portfolio)
            st.plotly_chart(weight_fig, use_container_width=True)

        with col2:
            sector_fig = plot_sector_distribution(portfolio)
            st.plotly_chart(sector_fig, use_container_width=True)

        st.subheader("Portfolio Composition")
        # Style the dataframe
        portfolio["Weight"] = portfolio["Weight"].astype(float) * 100
        st.dataframe(
            portfolio,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                "Company": st.column_config.TextColumn("Company", width="medium"),
                "Sector": st.column_config.TextColumn("Sector", width="medium"),
                "Strike Price ($)": st.column_config.NumberColumn(
                    "Strike Price ($)",
                    format="$%.2f",
                    width="small"
                ),
                "Weight": st.column_config.NumberColumn(
                    "Weight",
                    format="%.2f%%",
                    width="small",
                    help="Portfolio weight in percentage"
                ),
                "Model Score": st.column_config.NumberColumn(
                    "Model Score",
                    format="%.1f",
                    width="small"
                ),
            },
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.warning("No data available for the selected date.")


if __name__ == "__main__":
    main()
