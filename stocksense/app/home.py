import os
import subprocess
import streamlit as st


def run():
    """
    Runs home script.
    """

    st.set_page_config(
        layout="wide",
        page_title="Stocksense Home",
        page_icon="🏠",
    )

    st.sidebar.title("Stocksense App")
    st.sidebar.success("Select page")

    # Add pages links
    st.sidebar.page_link("home.py", label="Home", icon="🏠")
    st.sidebar.page_link("pages/explore.py", label="Explore Stock Data", icon="🌎")
    st.sidebar.page_link("pages/analytics.py", label="Stock Analytics", icon="📈")
    st.sidebar.page_link("pages/insights.py", label="Stock Picks", icon="🔮")
    st.sidebar.divider()

    st.markdown(
        """
        Welcome to Stocksense Analytics App.
        """
    )


def main():
    script_path = os.path.abspath(__file__)
    command = ["streamlit", "run", script_path]
    subprocess.run(command)


if __name__ == "__main__":
    run()
