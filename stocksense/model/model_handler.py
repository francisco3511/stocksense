import datetime as dt
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import polars as pl
from loguru import logger

from stocksense.config import config

from .optuna_optimizer import OptunaOptimizer
from .utils import (
    find_last_trading_date,
    format_xgboost_params,
    get_train_bounds,
    validate_trade_date,
)
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
        self.trade_date = trade_date if trade_date else find_last_trading_date()
        if not validate_trade_date(self.trade_date):
            raise ValueError(f"Invalid trade date: {self.trade_date}.")

    def train(self, data: pl.DataFrame, retrain: bool = False) -> None:
        """
        Train Optuna-XGBoost models for stock selection.

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

                start_date, end_date = get_train_bounds(
                    self.trade_date,
                    config.model.max_train_years
                )

                train = data.filter(
                    (pl.col("tdq") <= end_date) &
                    (~pl.all_horizontal(pl.col(target).is_null()))
                ).select(["tdq", "tic"] + self.features + [target])

                best_params = self._optimize(train, self.features, target)
                train = train.filter(pl.col("tdq") > start_date)
                final_params = format_xgboost_params(best_params, 100)

                logger.info(f"Training {target} model for {self.trade_date} with {len(train)} rows")
                logger.info(f"Training with params: {final_params}")

                X_train = train.select(self.features).to_pandas()
                y_train = train.select(target).to_pandas().values.ravel()
                model = XGBoostClassifier(final_params)
                model.train(X_train, y_train)
                model.save_model(model_file)
                logger.success(f"END training model for {target}, {self.trade_date}")
            return
        except Exception as e:
            logger.error(f"ERROR: failed to train model - {e}")
            raise

    def _optimize(
        self,
        train: pl.DataFrame,
        features: List[str],
        target: str
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
        """
        optimizer = OptunaOptimizer(n_trials=500)
        best_solution = optimizer.optimize(
            train,
            features,
            target,
            config.model.max_train_years
        )
        return best_solution

    def score(self, data: pl.DataFrame) -> None:
        """
        Score stocks using rank-based ensemble of target-specific models.

        Parameters
        ----------
        data : pl.DataFrame
            Preprocessed financial data.

        Returns
        -------
        pl.DataFrame
            Dataframe with stock ranks.
        """
        try:
            logger.info(f"START stocksense eval - {self.trade_date}")
            test = data.filter((pl.col("tdq") == self.trade_date))
            final_ranks = test.clone()
            pred_cols = []
            perc_cols = []

            # Score along each target
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
                n_bins = 100
                n_elements = len(prob_scores)
                final_ranks = final_ranks.with_columns([
                    pl.Series(f"pred_{target}", prob_scores),
                    (
                        pl.Series(f"pred_{target}", prob_scores)
                        .rank(method="ordinal", descending=False)
                        .map_elements(
                            lambda x, n=n_bins, total=n_elements: int(np.ceil(x * n / total))
                        )
                    ).alias(f"perc_{target}")
                ])
                pred_cols.append(f"pred_{target}")
                perc_cols.append(f"perc_{target}")


            final_ranks = final_ranks.with_columns(
                pl.mean_horizontal([pl.col(col) for col in perc_cols]).round(2).alias("avg_score")
            ).sort("avg_score", descending=True)

            report_cols = ["tic", "adj_close", "max_return_4Q", "fwd_return_4Q", "avg_score"]
            self._save_scoring_report(final_ranks.select(report_cols + pred_cols))

            return final_ranks
        except Exception as e:
            logger.error(f"ERROR: failed to score stocks - {e}")
            raise

    def _save_scoring_report(self, rank_data: pl.DataFrame) -> None:
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
