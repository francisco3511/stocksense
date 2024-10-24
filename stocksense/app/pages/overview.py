
import streamlit as st
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from database_handler import DatabaseHandler

MILLION = 1000000
MARGIN = dict(l=0, r=10, b=10, t=25)


def get_index_constituents():
    db = DatabaseHandler()
    stocks = db.fetch_stock().to_pandas()
    return stocks.loc[
        stocks.spx_status == 1,  # noqa: E712
    ]


def date_breaks(df, date_col='date'):
    dt_all = pd.date_range(start=df[date_col].min(), end=df[date_col].max())
    dt_obs = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(df[date_col])]
    return [d for d in dt_all.strftime("%Y-%m-%d").tolist() if d not in dt_obs]


@st.cache_data(show_spinner="Fetching index data...", max_entries=10)
def load_index_data():
    db = DatabaseHandler()
    return db.fetch_index_data().to_pandas()


@st.cache_data(show_spinner="Fetching data...", max_entries=10)
def load_data():
    db = DatabaseHandler()
    stock = db.fetch_stock().to_pandas()
    info = db.fetch_info().to_pandas()
    return stock, info


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

    #  stock, info = load_data()


if __name__ == "__main__":
    main()
