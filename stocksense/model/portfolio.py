import datetime as dt
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger

from stocksense.database_handler import DatabaseHandler


class PortfolioBuilder:
    """
    Portfolio construction class.
    Handles portfolio creation based on model predictions.
    """

    def __init__(
        self,
        n_stocks: int = 30,
        weighting: str = "market_cap",
        sector_constraints: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
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
        self.n_stocks = n_stocks
        self.weighting = weighting
        self.sector_constraints = sector_constraints or {}
        self.db = DatabaseHandler()
        self.scores_dir = Path(__file__).parents[2] / "reports" / "scores"
        self.portfolios_dir = Path(__file__).parents[2] / "reports" / "portfolios"

    def build_portfolio(self, trade_date: dt.datetime) -> pl.DataFrame:
        """
        Build portfolio based on model predictions.

        Parameters
        ----------
        trade_date : dt.datetime
            Trade date in YYYY-MM-DD format

        Returns
        -------
        pl.DataFrame
            Portfolio allocation dataframe
        """
        try:
            scores_file = self.scores_dir / f"scores_{trade_date.date()}.csv"
            if not scores_file.exists():
                raise FileNotFoundError(f"No report found for trade date {trade_date}")

            predictions = pl.read_csv(scores_file, columns=["tic", "adj_close", "pred"])
            stock_info = self.db.fetch_stock()
            stock_status = self.db.fetch_info()

            top_stocks = predictions.head(self.n_stocks)
            portfolio = top_stocks.join(stock_info.select(["tic", "name", "sector"]), on="tic")
            portfolio = portfolio.join(stock_status.select(["tic", "market_cap"]), on="tic")

            if self.weighting == "equal":
                weights = self._equal_weight(portfolio)
            elif self.weighting == "market_cap":
                weights = self._market_cap_weight(portfolio)
            elif self.weighting == "sector_neutral":
                weights = self._sector_neutral_weight(portfolio, trade_date)
            else:
                raise ValueError(f"Unknown weighting scheme: {self.weighting}")

            if self.sector_constraints:
                weights = self._apply_sector_constraints(portfolio, weights)

            portfolio = portfolio.with_columns(pl.Series("weight", weights))
            portfolio = portfolio.select(["tic", "name", "sector", "pred", "adj_close", "weight"])

            logger.info(f"Built {self.weighting}-weighted portfolio with {self.n_stocks} stocks")
            self._save_portfolio_excel(portfolio, trade_date)
            return portfolio.sort("weight", descending=True)

        except Exception as e:
            logger.error(f"Failed to build portfolio: {e}")
            raise

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

        # Apply sector-based weights to individual stocks
        weights = np.array([sector_weights[sector] for sector in portfolio["sector"]])
        return weights / weights.sum()

    def _market_cap_weight(self, portfolio: pl.DataFrame) -> np.ndarray:
        """
        Market cap weighting scheme.
        Weights are proportional to each stock's market cap relative to
        the total market cap of selected stocks.

        Parameters
        ----------
        portfolio : pl.DataFrame
            Portfolio dataframe.

        Returns
        -------
        np.ndarray
            Market cap weights.
        """

        market_caps = portfolio["market_cap"].to_numpy()
        weights = market_caps / market_caps.sum()
        return weights

    def _apply_sector_constraints(self, portfolio: pl.DataFrame, weights: np.ndarray) -> np.ndarray:
        """
        Apply sector allocation constraints.

        Parameters
        ----------
        portfolio : pl.DataFrame
            Portfolio dataframe.
        weights : np.ndarray
            Weights to apply constraints to.

        Returns
        -------
        np.ndarray
            Constrained weights.
        """

        for sector, (min_weight, max_weight) in self.sector_constraints.items():
            sector_mask = portfolio["sector"] == sector
            sector_weight = weights[sector_mask].sum()

            if sector_weight < min_weight:
                scale = min_weight / sector_weight
                weights[sector_mask] *= scale
                weights[~sector_mask] *= (1 - min_weight) / (1 - sector_weight)
            elif sector_weight > max_weight:
                scale = max_weight / sector_weight
                weights[sector_mask] *= scale
                weights[~sector_mask] *= (1 - max_weight) / (1 - sector_weight)

        return weights / weights.sum()

    def _log_portfolio(self, portfolio: pl.DataFrame) -> None:
        """
        Log portfolio details in a formatted way.

        Parameters
        ----------
        portfolio : pl.DataFrame
            Portfolio dataframe.
        """

        logger.info("\nPortfolio Summary:")
        logger.info("-" * 80)

        sector_alloc = (
            portfolio.group_by("sector").agg(pl.col("weight").sum()).sort("weight", descending=True)
        )
        logger.info("Sector Allocations:")
        for row in sector_alloc.iter_rows():
            logger.info(f"  {row[0]:<20} {row[1]:>7.2%}")

        top_positions = portfolio.sort("weight", descending=True).head(self.n_stocks)
        logger.info("\nTop 10 Positions:")
        for row in top_positions.iter_rows():
            logger.info(f"  {row[0]:<6} {row[3]:<20} {row[5]:>7.2%}")

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
            portfolio.sort("weight", descending=True).to_pandas().to_excel(
                writer, sheet_name="Full Portfolio", index=False
            )
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
            top_positions = portfolio.sort("weight", descending=True).head(10).to_pandas()
            top_positions["weight"] = top_positions["weight"].map("{:.2%}".format)
            top_positions.to_excel(writer, sheet_name="Top Holdings", index=False)

        logger.info(f"Portfolio details saved to {excel_path}")
