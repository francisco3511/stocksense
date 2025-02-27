import datetime as dt

import pandas as pd
import plotly.express as px
import streamlit as st

from stocksense.config import PORTFOLIO_DIR, SCORES_DIR
from stocksense.database import DatabaseHandler


@st.cache_data(show_spinner="Loading stock data...", max_entries=10)
def load_stock_data():
    db = DatabaseHandler()
    return db.fetch_stock().to_pandas()


def get_available_portfolios():
    """
    Get all available portfolio files.

    Returns
    -------
    dates : list
        List of dates from the portfolio files.
    """
    portfolio_files = list(PORTFOLIO_DIR.glob("portfolio_*.xlsx"))
    dates = [dt.datetime.strptime(f.stem.split("_")[1], "%Y-%m-%d").date() for f in portfolio_files]
    return sorted(dates, reverse=True)


def load_portfolio(trade_date):
    """
    Load portfolio for a specific trade date.

    Parameters
    ----------
    trade_date : dt.date
        Trade date to load portfolio for.

    Returns
    -------
    portfolio_file : Path
        Path to the portfolio file.
    """
    portfolio_file = PORTFOLIO_DIR / f"portfolio_{trade_date}.xlsx"
    if not portfolio_file.exists():
        st.error(f"No portfolio found for trade date {trade_date}")
        return None
    return pd.read_excel(portfolio_file)


def load_scoring_report(trade_date):
    """
    Load scoring report for a specific trade date.

    Parameters
    ----------
    trade_date : dt.date
        Trade date to load scoring report for.

    Returns
    -------
    scoring_report_file : Path
        Path to the scoring report file.
    """
    scoring_report_file = SCORES_DIR / f"scores_{trade_date}.csv"
    return pd.read_csv(scoring_report_file)


def plot_sector_distribution(portfolio_data):
    """
    Plot sector distribution of selected stocks.

    Parameters
    ----------
    portfolio_data : pd.DataFrame
        Portfolio data for the portfolio.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure for the sector distribution.
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

    Parameters
    ----------
    portfolio_data : pd.DataFrame
        Portfolio data for the portfolio.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure for the weight distribution.
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


def calculate_forward_returns(scoring_report, portfolio_data):
    """
    Calculate portfolio and SP500 forward returns.

    Parameters
    ----------
    scoring_report : pd.DataFrame
        Scoring report for the portfolio.
    portfolio_data : pd.DataFrame
        Portfolio data for the portfolio.

    Returns
    -------
    (portfolio_return, sp500_return, max_portfolio_return) or None if data not available.
    """
    if "Forward Return 1Y" not in portfolio_data.columns:
        return None

    fwd_returns = portfolio_data["Forward Return 1Y"]
    max_return = portfolio_data["Max Return 1Y"]
    portfolio_return = (portfolio_data["Weight"] * (fwd_returns / 100)).sum()
    max_portfolio_return = (portfolio_data["Weight"] * (max_return / 100)).sum()
    sp500_return = scoring_report["avg_index_fwd_return_4Q"].iloc[0]
    return portfolio_return, sp500_return, max_portfolio_return


def calculate_probability_distribution(scoring_report):
    """
    Calculate probability distribution of model scores.

    Parameters
    ----------
    scoring_report : pd.DataFrame
        Scoring report for the portfolio.

    Returns
    -------
    bounds : list
        List of tuples containing the probability column name, minimum value, and maximum value.
    """
    bounds = []
    for prob_col in [col for col in scoring_report.columns if col.startswith("pred_")]:
        bounds.append((prob_col, scoring_report[prob_col].min(), scoring_report[prob_col].max()))
    return bounds


def display_portfolio_metrics(scoring_report, portfolio_data):
    """
    Display key portfolio metrics.

    Parameters
    ----------
    scoring_report : pd.DataFrame
        Scoring report for the portfolio.
    portfolio_data : pd.DataFrame
        Portfolio data for the portfolio.
    """

    portfolio_data['Weight'] = portfolio_data['Weight'].str.rstrip('%').astype(float) / 100
    st.subheader("Model Target Analysis")
    pred_cols = [col for col in scoring_report.columns if col.startswith('pred_')]

    for i in range(0, len(pred_cols), 3):
        cols = st.columns(4)
        for j, col in enumerate(pred_cols[i:i+3]):
            if col == 'pred_aggressive_hit':
                label = "Aggressive Target"
            elif col == 'pred_moderate_hit':
                label = "Moderate Target"
            elif col == 'pred_relaxed_hit':
                label = "Relaxed Target"
            else:
                label = col.replace('pred_', '').replace('_', ' ').title()

            with cols[j]:
                st.metric(
                    label,
                    f"{scoring_report[col].mean():.1%}",
                    help=(
                        f"Range: {scoring_report[col].min():.1%} to "
                        f"{scoring_report[col].max():.1%}"
                    )
                )

    forward_returns = calculate_forward_returns(scoring_report, portfolio_data)
    if forward_returns:
        st.subheader("Portfolio Performance")
        portfolio_return, sp500_return, max_portfolio_return = forward_returns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Upside", f"{max_portfolio_return:.1%}")
        with col2:
            st.metric("Portfolio 1Y Return", f"{portfolio_return:.1%}")
        with col3:
            st.metric("S&P500 1Y Return", f"{sp500_return:.1%}")
        with col4:
            alpha = portfolio_return - sp500_return
            st.metric("Alpha", f"{alpha:.1%}")


def main():
    """Insights main script."""
    st.set_page_config(layout="wide", page_title="Portfolio Insights", page_icon="💼")

    # Sidebar navigation
    st.sidebar.title("Stocksense App")
    st.sidebar.divider()
    st.sidebar.success("Select page")
    st.sidebar.page_link("home.py", label="Home", icon="🏠")
    st.sidebar.page_link("pages/overview.py", label="Market Overview", icon="🌎")
    st.sidebar.page_link("pages/analytics.py", label="Stock Analytics", icon="📈")
    st.sidebar.page_link("pages/insights.py", label="Portfolio Insights", icon="💼")
    st.sidebar.divider()

    # Main content
    st.title("Portfolio Insights 💼")
    st.write("")

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
    report = load_scoring_report(trade_date)

    if portfolio is not None:
        st.write("")
        display_portfolio_metrics(report, portfolio)
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            weight_fig = plot_weight_distribution(portfolio)
            st.plotly_chart(weight_fig, use_container_width=True)

        with col2:
            sector_fig = plot_sector_distribution(portfolio)
            st.plotly_chart(sector_fig, use_container_width=True)

        st.subheader("Portfolio Composition")

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
