import datetime as dt
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from stocksense.database import DatabaseHandler

REPORTS_DIR = Path(__file__).parents[3] / "reports"
SCORES_DIR = REPORTS_DIR / "scores"
PORTFOLIOS_DIR = REPORTS_DIR / "portfolios"


@st.cache_data(show_spinner="Loading stock data...", max_entries=10)
def load_stock_data():
    db = DatabaseHandler()
    return db.fetch_stock().to_pandas()


def get_available_dates():
    """
    Get all available trade dates from score files.
    """
    score_files = list(SCORES_DIR.glob("scores_*.csv"))
    dates = [dt.datetime.strptime(f.stem.split("_")[1], "%Y-%m-%d").date() for f in score_files]
    return sorted(dates, reverse=True)


def load_scores(trade_date):
    """
    Load scores for a specific trade date.
    """
    score_file = SCORES_DIR / f"scores_{trade_date}.csv"
    if not score_file.exists():
        st.error(f"No scores found for trade date {trade_date}")
        return None
    return pd.read_csv(score_file)


def plot_sector_distribution(portfolio_data):
    """
    Plot sector distribution of selected stocks.
    """
    sector_dist = portfolio_data.groupby("sector")["weight"].sum().reset_index()
    fig = px.pie(
        sector_dist,
        values="weight",
        names="sector",
        title="Sector Distribution",
        template="plotly_dark",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)


def display_portfolio_metrics(portfolio_data):
    """
    Display key portfolio metrics.
    """
    total_stocks = len(portfolio_data)
    avg_score = portfolio_data["pred"].mean()
    avg_price = portfolio_data["adj_close"].mean()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Stocks", total_stocks)
    with col2:
        st.metric("Average Model Score", f"{avg_score:.3f}")
    with col3:
        st.metric("Average Stock Price", f"${avg_price:.2f}")


def main():
    """Insights main script."""

    st.set_page_config(layout="wide", page_title="Stock Picks", page_icon="🔮")
    st.sidebar.title("Stocksense App")
    st.sidebar.success("Select page")

    st.sidebar.page_link("home.py", label="Home", icon="🏠")
    st.sidebar.page_link("pages/overview.py", label="Market Overview", icon="🌎")
    st.sidebar.page_link("pages/analytics.py", label="Stock Analytics", icon="📈")
    st.sidebar.page_link("pages/insights.py", label="Stock Picks", icon="🔮")
    st.sidebar.divider()

    st.title("Stock Picks Insights")

    stock_data = load_stock_data()
    available_dates = get_available_dates()
    trade_date = st.selectbox("Select Trade Date", available_dates)
    scores = load_scores(trade_date)
    scores = scores.join(stock_data, on="tic", rsuffix="_stock")

    if scores is not None:
        display_portfolio_metrics(scores)
        plot_sector_distribution(scores)

        st.subheader("Top 30 Selected Stocks")
        columns_to_display = ["symbol", "company_name", "sector", "pred", "adj_close", "weight"]
        formatted_scores = scores[columns_to_display].head(30)

        formatted_scores["pred"] = formatted_scores["pred"].round(3)
        formatted_scores["adj_close"] = formatted_scores["adj_close"].round(2)
        formatted_scores["weight"] = (formatted_scores["weight"] * 100).round(2).astype(str) + "%"
        formatted_scores.columns = ["Symbol", "Company", "Sector", "Score", "Price ($)", "Weight"]

        st.dataframe(formatted_scores, use_container_width=True)
    else:
        st.warning("No data available for the selected date.")


if __name__ == "__main__":
    main()
