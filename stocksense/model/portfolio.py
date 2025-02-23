import datetime as dt

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger

from stocksense.config import PORTFOLIO_DIR
from stocksense.database import DatabaseHandler


class PortfolioBuilder:
    """
    Portfolio construction class.
    Handles portfolio creation based on model predictions.
    """

    def __init__(self, weighting: str = "market_cap"):
        """
        Initialize portfolio builder.

        Parameters
        ----------
        n_stocks : int
            Number of stocks to include in portfolio
        weighting : str
            Weighting scheme ('equal', 'market_cap', or 'sector_neutral')
        """
        self.weighting = weighting
        self.db = DatabaseHandler()

    def build_portfolio(
        self,
        n_stocks: int,
        trade_date: dt.datetime,
        data: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Build portfolio based on model predictions.

        Parameters
        ----------
        n_stocks : int
            Number of stocks to include in portfolio
        trade_date : dt.datetime
            Trade date in YYYY-MM-DD format
        data : pl.DataFrame
            Dataframe with scored stock data

        Returns
        -------
        pl.DataFrame
            Portfolio allocation dataframe
        """
        try:
            stock_info = self.db.fetch_stock()
            data = data.sort("avg_score", descending=True)
            scored_stocks = data.join(
                stock_info.select(["tic", "name", "sector"]), on="tic", how="left"
            )

            portfolio = self._filter_candidates(scored_stocks)

            if self.weighting == "equal":
                weights = self._equal_weight(portfolio)
            elif self.weighting == "market_cap":
                weights = self._market_cap_weight(portfolio, score_weight=0.7)
            elif self.weighting == "sector_neutral":
                weights = self._sector_neutral_weight(portfolio, trade_date)
            else:
                raise ValueError(f"Unknown weighting scheme: {self.weighting}")

            portfolio = portfolio.with_columns(pl.Series("weight", weights))
            portfolio_cols = [
                "tic", "name", "sector",
                "adj_close", "mkt_cap",
                "avg_score", "weight"
            ]

            if trade_date > dt.datetime.now() - dt.timedelta(days=365):
                portfolio = portfolio.select(portfolio_cols)
            else:
                portfolio = portfolio.select(portfolio_cols + ["max_return_4Q", "fwd_return_4Q"])

            logger.info(f"Built {self.weighting}-weighted portfolio with {n_stocks} stocks")
            portfolio = portfolio.sort("weight", descending=True).head(n_stocks)
            self._save_portfolio_excel(portfolio, trade_date)
            return portfolio

        except Exception as e:
            logger.error(f"Failed to build portfolio: {e}")
            raise

    def _filter_candidates(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply fundamental and technical filters to screen stocks.

        Parameters
        ----------
        df : pl.DataFrame
            Filtered portfolio.
        """
        return df.filter(
            (pl.col("saleq_yoy") > -30) &
            (pl.col("price_mom") > -25) &
            (pl.col("avg_score") > 80)
        )

    def _equal_weight(self, portfolio: pl.DataFrame) -> np.ndarray:
        """Equal weighting scheme."""
        return np.ones(len(portfolio)) / len(portfolio)

    def _sector_neutral_weight(
        self,
        portfolio: pl.DataFrame,
        trade_date: dt.datetime
    ) -> np.ndarray:
        """
        Sector-neutral weighting scheme.

        Parameters
        ----------
        portfolio : pl.DataFrame
            Portfolio dataframe.
        trade_date : dt.datetime
            Trade date.

        Returns
        -------
        np.ndarray
            Sector-neutral weights.
        """

        sp500_sectors = (
            self.db.fetch_stock()
            .filter(
                (pl.col("date_removed").is_null() | (pl.col("date_removed") > trade_date))
                & (pl.col("date_added").is_null() | (pl.col("date_added") <= trade_date))
            )
            .group_by("sector")
            .count()
            .with_columns((pl.col("count") / pl.col("count").sum()).alias("sector_weight"))
        )
        portfolio_sectors = portfolio.group_by("sector").count()
        sector_weights = {}
        for sector in portfolio_sectors["sector"]:
            sector_target = sp500_sectors.filter(pl.col("sector") == sector)["sector_weight"][0]
            sector_count = portfolio_sectors.filter(pl.col("sector") == sector)["count"][0]
            sector_weights[sector] = sector_target / sector_count

        weights = np.array([sector_weights[sector] for sector in portfolio["sector"]])
        return weights / weights.sum()

    def _market_cap_weight(self, portfolio: pl.DataFrame, score_weight: float = 0.5) -> np.ndarray:
        """
        Hybrid market cap and score weighting scheme using rank-based normalization.

        Parameters
        ----------
        portfolio : pl.DataFrame
            Portfolio dataframe
        score_weight : float
            Weight given to the model score (between 0 and 1)
            0 = pure market cap weighting
            1 = pure rank weighting

        Returns
        -------
        np.ndarray
            Blended weights
        """
        market_caps = portfolio["mkt_cap"].to_numpy()
        mkt_weights = market_caps / market_caps.sum()

        scores = portfolio["avg_score"].to_numpy()
        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
        score_weights = normalized_scores / normalized_scores.sum()

        final_weights = (1 - score_weight) * mkt_weights + score_weight * score_weights
        return final_weights / final_weights.sum()

    def _save_portfolio_excel(self, portfolio: pl.DataFrame, trade_date: dt.datetime) -> None:
        """
        Save portfolio details to a multi-sheet Excel file.

        Parameters
        ----------
        portfolio : pl.DataFrame
            Portfolio dataframe.
        trade_date : dt.datetime
            Trade date.
        """
        excel_path = PORTFOLIO_DIR / f"portfolio_{trade_date.date()}.xlsx"
        portfolio_pd = portfolio.rename({
            "tic": "Ticker",
            "name": "Company",
            "sector": "Sector",
            "adj_close": "Strike Price ($)",
            "mkt_cap": "Market Cap ($M)",
            "avg_score": "Model Score",
            "weight": "Weight",
            "max_return_4Q": "Max Return 1Y",
            "fwd_return_4Q": "Forward Return 1Y"
        }, strict=False).to_pandas()

        portfolio_pd["Weight"] = portfolio_pd["Weight"].map("{:.2%}".format)
        portfolio_pd["Model Score"] = portfolio_pd["Model Score"].round(2)
        portfolio_pd["Market Cap ($M)"] = portfolio_pd["Market Cap ($M)"].round(2)
        portfolio_pd["Strike Price ($)"] = portfolio_pd["Strike Price ($)"].round(1)
        if "Max Return 1Y" in portfolio_pd.columns:
            portfolio_pd["Max Return 1Y"] = portfolio_pd["Max Return 1Y"].round(2)
            portfolio_pd["Forward Return 1Y"] = portfolio_pd["Forward Return 1Y"].round(2)

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            # Sheet 1: Full Portfolio
            portfolio_pd.sort_values("Weight", ascending=False).to_excel(
                writer, sheet_name="Full Portfolio", index=False
            )

            # Sheet 2: Sector Allocations
            portfolio_numeric = portfolio_pd.copy()
            portfolio_numeric["Weight"] = portfolio_pd["Weight"].str.rstrip('%').astype(float) / 100
            sector_alloc = (
                portfolio_numeric.groupby("Sector")["Weight"]
                .sum()
                .reset_index()
                .sort_values("Weight", ascending=False)
            )
            sector_alloc["Weight"] = sector_alloc["Weight"].map("{:.2%}".format)
            sector_alloc.to_excel(writer, sheet_name="Sector Allocations", index=False)

            # Sheet 3: Top Holdings
            top_positions = portfolio_pd.sort_values("Weight", ascending=False).head(5)
            top_positions.to_excel(writer, sheet_name="Top Holdings", index=False)

        logger.info(f"Portfolio saved to {excel_path}")
