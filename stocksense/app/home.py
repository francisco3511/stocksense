import os
import subprocess

import streamlit as st


def run():
    """
    Runs home script.
    """

    st.set_page_config(
        layout="wide",
        page_title="Stocksense",
        page_icon="📈",
        initial_sidebar_state="expanded",
    )
    st.sidebar.title("Stocksense App")
    st.sidebar.success("Select page")

    st.sidebar.page_link("home.py", label="Home", icon="🏠")
    st.sidebar.page_link("pages/overview.py", label="Market Overview", icon="🌎")
    st.sidebar.page_link("pages/analytics.py", label="Stock Analytics", icon="📈")
    st.sidebar.page_link("pages/insights.py", label="Stock Picks", icon="💼")
    st.sidebar.divider()

    st.header("Welcome to Stocksense Analytics App!")
    st.divider()

    # Overview section
    st.subheader("📊 App Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌎 Market Overview")
        st.markdown("""
        - Real-time S&P 500 market summary
        - Sector distribution analysis
        - Latest earnings reports tracking
        - Key market metrics and statistics
        """)

        st.markdown("### 📈 Stock Analytics")
        st.markdown("""
        - Detailed individual stock analysis
        - Technical and fundamental indicators
        - Financial statements analysis
        - Insider trading tracking
        """)

    with col2:
        st.markdown("### 💼 Portfolio Insights")
        st.markdown("""
        - Model-driven stock selection
        - Portfolio composition analysis
        - Sector allocation visualization
        - Performance metrics tracking
        """)

        st.markdown("### 🔍 Key Features")
        st.markdown("""
        - Interactive data visualization
        - Accurate market data
        - Curated financial metrics
        - User-friendly interface
        """)


def main():
    script_path = os.path.abspath(__file__)
    command = ["streamlit", "run", script_path]
    subprocess.run(command)


if __name__ == "__main__":
    run()
