import datetime as dt
import warnings
from pathlib import Path
from typing import List, Optional

import polars as pl
from loguru import logger

from stocksense.config import config

from .genetic_algorithm import GeneticAlgorithm, fitness_function_wrapper
from .utils import find_last_trading_date, format_xgboost_params, validate_trade_date
from .xgboost_model import XGBoostClassifier

MODEL_DIR = Path(__file__).parents[1] / "model" / "model_base"
REPORT_DIR = Path(__file__).parents[2] / "reports" / "scores"

warnings.filterwarnings("ignore")


class ModelHandler:
    """
    Stocksense stock selection model handling class.
    Basic handling for model training, evaluation and testing.
    """

    def __init__(self, trade_date: Optional[dt.datetime] = None):
        self.features = config.model.features
        self.targets = config.model.targets
        self.prediction_horizon = config.processing.prediction_horizon
        self.min_train_years = config.model.min_train_years
        self.trade_date = trade_date if trade_date else find_last_trading_date()
        if not validate_trade_date(self.trade_date):
            raise ValueError(f"Invalid trade date: {self.trade_date}.")

    def train(self, data: pl.DataFrame, retrain: bool = False) -> None:
        """
        Train GA-XGBoost models for stock selection.

        Parameters
        ----------
        data : pl.DataFrame
            Preprocessed financial data.
        retrain : bool
            Whether to retrain the model for given trade date.
        """
        try:
            for target in self.targets:
                logger.info(f"START training model for {target}, {self.trade_date}")
                trade_date_model_dir = MODEL_DIR / f"{self.trade_date.date()}"
                trade_date_model_dir.mkdir(parents=True, exist_ok=True)
                model_file = trade_date_model_dir / f"{target}.pkl"

                if model_file.exists() and not retrain:
                    logger.warning(f"Model already exists for {target}, {self.trade_date}")
                    continue

                train = data.filter(
                    (pl.col("tdq") < self.trade_date - dt.timedelta(days=360))
                    & ~pl.all_horizontal(pl.col(target).is_null())
                ).select(["tdq", "tic"] + self.features + [target])

                scale = self.get_dataset_imbalance_scale(train, target)
                logger.info(f"Class balance scale: {scale}")

                params = self.optimize(train, self.features, target, scale, self.min_train_years)
                params = format_xgboost_params(params, scale)

                X_train = train.select(self.features).to_pandas()
                y_train = train.select(target).to_pandas().values.ravel()

                model = XGBoostClassifier(params)
                model.train(X_train, y_train)
                model.save_model(model_file)
                logger.success(f"END training model for {target}, {self.trade_date}")
            return
        except Exception as e:
            logger.error(f"ERROR: failed to train model - {e}")
            raise

    def optimize(
        self,
        train: pl.DataFrame,
        features: List[str],
        target: str,
        scale: float,
        min_train_years: int,
    ) -> None:
        """
        Optimize model parameters.

        Parameters
        ----------
        train : pl.DataFrame
            Preprocessed financial data.
        features : List[str]
            List of features to use for model optimization.
        target : str
            Target variable to optimize model for.
        scale : float
            Class imbalance scale.
        min_train_years : int
            Minimum number of years to use for training.
        """
        ga = GeneticAlgorithm(
            ga_settings=config.model.ga,
            fitness_func=fitness_function_wrapper(train, features, target, scale, min_train_years),
        )
        ga.create_instance()
        ga.train()
        best_solution, best_solution_fitness, best_solution_idx = ga.best_solution()
        return best_solution

    def score(self, data: pl.DataFrame, stocks: list[str]) -> None:
        """
        Score stocks using rank-based ensemble of target-specific models.

        Parameters
        ----------
        data : pl.DataFrame
            Preprocessed financial data.
        stocks : list[str]
            List of stocks to score.

        Returns
        -------
        pl.DataFrame
            Dataframe with stock ranks.
        """
        try:
            logger.info(f"START stocksense eval - {self.trade_date}")
            test = data.filter((pl.col("tdq") == self.trade_date) & pl.col("tic").is_in(stocks))
            final_ranks = test.clone()
            pred_cols = []

            weights = self.get_market_regime_weights(data)

            # Get predictions and convert to ranks for each target
            for target in self.targets:
                trade_date_model_dir = MODEL_DIR / f"{self.trade_date.date()}"
                model_file = trade_date_model_dir / f"{target}.pkl"
                if not model_file.exists():
                    raise FileNotFoundError(f"No model found for trade date {self.trade_date}")

                model = XGBoostClassifier()
                model.load_model(model_file)
                logger.info(f"loaded model from {model_file}, with params: {model.params}")

                test_df = test.select(self.features).to_pandas()
                prob_scores = model.predict_proba(test_df)

                final_ranks = final_ranks.with_columns(
                    [
                        pl.Series(f"pred_{target}", prob_scores),
                        (
                            (
                                pl.lit(prob_scores)
                                .rank("ordinal", descending=True)
                                .cast(pl.Float64)
                                / len(stocks)
                            ).alias(f"rank_{target}")
                        ),
                    ]
                )
                pred_cols.append(f"pred_{target}")

            # Calculate weighted average rank
            weighted_ranks = [pl.col(f"rank_{target}") * weights[target] for target in self.targets]
            final_ranks = final_ranks.with_columns(
                pl.sum_horizontal(weighted_ranks).alias("avg_rank")
            ).sort("avg_rank")

            report_cols = ["tic", "adj_close", "max_return_4Q", "fwd_return_4Q", "avg_rank"]
            self.save_scoring_report(final_ranks.select(report_cols + pred_cols))

            return final_ranks
        except Exception as e:
            logger.error(f"ERROR: failed to score stocks - {e}")
            raise

    def get_dataset_imbalance_scale(self, train: pl.DataFrame, target: str) -> float:
        """
        Compute dataset class imbalance scale.

        Parameters
        ----------
        train : pl.DataFrame
            Training dataset.
        target : str
            Target variable to compute class imbalance scale for.

        Returns
        -------
        float
            Class imbalance scale.
        """
        neg_count = len(train.filter(pl.col(target) == 0))
        pos_count = len(train.filter(pl.col(target) == 1))
        return max(1.0, round(neg_count / pos_count, 2))

    def save_scoring_report(self, rank_data: pl.DataFrame) -> None:
        """
        Save scoring report csv with ranks for each target and average rank.

        Parameters
        ----------
        rank_data : pl.DataFrame
            DataFrame containing ranks for each target and average rank.
        """
        try:
            report_file = REPORT_DIR / f"scores_{self.trade_date.date()}.csv"
            rank_data.write_csv(report_file)
            logger.success(f"SAVED scoring report to {report_file}")
        except Exception as e:
            logger.error(f"ERROR failed to save scoring report - {e}")
            raise

    def get_market_regime_weights(self, data: pl.DataFrame) -> dict:
        """
        Determine market regime and return appropriate target weights.

        Parameters
        ----------
        data : pl.DataFrame
            Training data.

        Returns
        -------
        dict
            Dictionary with target weights.
        """

        market_state = data.filter(pl.col("tdq") == self.trade_date).select("index_qoq").row(0)[0]

        BULL_THRESHOLD = 5.0
        BEAR_THRESHOLD = -5.0

        if market_state > BULL_THRESHOLD:
            weights = {"aggressive_hit": 0.4, "balanced_hit": 0.3, "relaxed_hit": 0.3}
        elif market_state < BEAR_THRESHOLD:
            weights = {"aggressive_hit": 0.33, "balanced_hit": 0.34, "relaxed_hit": 0.33}
        else:
            weights = {"aggressive_hit": 0.33, "balanced_hit": 0.34, "relaxed_hit": 0.33}

        logger.info(f"Market regime: {market_state:.2f}% QoQ")
        logger.info(f"Model weights: {weights}")
        return weights
