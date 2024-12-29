import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger

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
        sector_constraints : Dict[str, Tuple[float, float]], optional
            Min/max allocation constraints per sector
        """
        self.weighting = weighting
        self.db = DatabaseHandler()
        self.portfolios_dir = Path(__file__).parents[2] / "reports" / "portfolios"

    def build_portfolio(
        self, n_stocks: int, trade_date: dt.datetime, data: pl.DataFrame
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

            # Apply filters to get qualified stocks
            portfolio = self._filter_candidates(scored_stocks).head(50)

            if self.weighting == "equal":
                weights = self._equal_weight(portfolio)
            elif self.weighting == "market_cap":
                weights = self._market_cap_weight(portfolio, score_weight=0.5)
            elif self.weighting == "sector_neutral":
                weights = self._sector_neutral_weight(portfolio, trade_date)
            else:
                raise ValueError(f"Unknown weighting scheme: {self.weighting}")

            portfolio = portfolio.with_columns(pl.Series("weight", weights))
            portfolio_cols = ["tic", "name", "sector", "adj_close", "avg_score", "weight"]

            if trade_date > dt.datetime.now() - dt.timedelta(days=365):
                portfolio = portfolio.select(portfolio_cols)
            else:
                portfolio = portfolio.select(portfolio_cols + ["fwd_return_4Q"])

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

        score_threshold = df["avg_score"].mean() + df["avg_score"].std()
        quality_filters = (
            (pl.col("pe") > 0)
            & (pl.col("saleq_yoy") > -20)
            & (pl.col("fcf_yoy") > -50)
            & (pl.col("price_mom") > -25)
            & (pl.col("avg_score") > score_threshold)
        )
        return df.filter(quality_filters)

    def _equal_weight(self, portfolio: pl.DataFrame) -> np.ndarray:
        """Equal weighting scheme."""
        return np.ones(len(portfolio)) / len(portfolio)

    def _sector_neutral_weight(
        self, portfolio: pl.DataFrame, trade_date: dt.datetime
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

    def _market_cap_weight(self, portfolio: pl.DataFrame, score_weight: float = 0.3) -> np.ndarray:
        """
        Hybrid market cap and score weighting scheme.
        Weights are calculated as a combination of market cap and model scores.

        Parameters
        ----------
        portfolio : pl.DataFrame
            Portfolio dataframe
        score_weight : float
            Weight given to the model score (between 0 and 1)
            0 = pure market cap weighting
            1 = pure score weighting

        Returns
        -------
        np.ndarray
            Blended weights
        """

        # Calculate market cap component
        market_caps = portfolio["mkt_cap"].to_numpy()
        mkt_weights = market_caps / market_caps.sum()

        # Calculate score component with exponential scaling
        scores = portfolio["avg_score"].to_numpy()
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min())
        score_component = norm_scores**2
        score_component = score_component / score_component.sum()

        # Blend the weights
        final_weights = (1 - score_weight) * mkt_weights + score_weight * score_component
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
        excel_path = self.portfolios_dir / f"portfolio_{trade_date.date()}.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            # Sheet 1: Full Portfolio
            portfolio_df = portfolio.sort("weight", descending=True).to_pandas()
            portfolio_df["weight"] = portfolio_df["weight"].map("{:.2%}".format)
            portfolio_df["avg_score"] = portfolio_df["avg_score"].round(3)
            portfolio_df["adj_close"] = portfolio_df["adj_close"].round(1)
            if "fwd_return_4Q" in portfolio_df.columns:
                portfolio_df["fwd_return_4Q"] = portfolio_df["fwd_return_4Q"].round(2)
            portfolio_df.to_excel(writer, sheet_name="Full Portfolio", index=False)

            # Sheet 2: Sector Allocations
            sector_alloc = (
                portfolio.group_by("sector")
                .agg(pl.col("weight").sum())
                .sort("weight", descending=True)
                .to_pandas()
            )
            sector_alloc["weight"] = sector_alloc["weight"].map("{:.2%}".format)
            sector_alloc.to_excel(writer, sheet_name="Sector Allocations", index=False)

            # Sheet 3: Top Holdings
            top_positions = portfolio.sort("weight", descending=True).head(5).to_pandas()
            top_positions["weight"] = top_positions["weight"].map("{:.2%}".format)
            top_positions.to_excel(writer, sheet_name="Top Holdings", index=False)

        logger.info(f"Portfolio saved to {excel_path}")
