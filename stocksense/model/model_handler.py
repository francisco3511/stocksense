import datetime as dt
import warnings
from pathlib import Path
from typing import List, Optional

import polars as pl
from loguru import logger

from stocksense.config import config

from .genetic_algorithm import GeneticAlgorithm, fitness_function_wrapper
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
                params = self.optimize(train, target, scale)
                params = format_ga_parameters(params, scale)

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

    def optimize(self, train: pl.DataFrame, target: str, scale: float) -> None:
        """
        Optimize model parameters.

        Parameters
        ----------
        train : pl.DataFrame
            Preprocessed financial data.
        target : str
            Target variable to optimize model for.
        scale : float
            Class imbalance scale.
        """

        ga = GeneticAlgorithm(
            ga_settings=config.model.ga,
            fitness_func=fitness_function_wrapper(
                train, self.features, target, scale, self.min_train_years
            ),
        )
        ga.create_instance()
        ga.train()
        best_solution, best_solution_fitness, best_solution_idx = ga.best_solution()
        return best_solution

    def score(self, data: pl.DataFrame, stocks: list[str]) -> None:
        """
        Score stocks using all target-specific models and save average ranks.

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

            final_ranks = data.filter(
                (pl.col("tdq") == self.trade_date) & pl.col("tic").is_in(stocks)
            )

            pred_cols = []
            for target in self.targets:
                trade_date_model_dir = MODEL_DIR / f"{self.trade_date.date()}"
                model_file = trade_date_model_dir / f"{target}.pkl"
                if not model_file.exists():
                    raise FileNotFoundError(f"No model found for trade date {self.trade_date}")

                test_df = (
                    data.filter((pl.col("tdq") == self.trade_date) & pl.col("tic").is_in(stocks))
                    .select(self.features)
                    .to_pandas()
                )

                model = XGBoostClassifier()
                model.load_model(model_file)
                logger.info(f"loaded model from {model_file}, with params: {model.params}")

                prob_scores = model.predict_proba(test_df)
                final_ranks = final_ranks.with_columns(pl.Series(f"pred_{target}", prob_scores))
                pred_cols.append(f"pred_{target}")

            # Calculate average rank
            final_ranks = (
                final_ranks.with_columns(pl.mean_horizontal(pred_cols).alias("avg_score"))
                .sort("avg_score", descending=True)
                .with_columns(pl.col("avg_score").round(4).alias("avg_score"))
            )

            self.save_scoring_report(
                final_ranks.select(["tic", "adj_close", "fwd_return_4Q", "avg_score"] + pred_cols)
            )

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
        min_year = pl.col("tdq").dt.year().min()
        filtered_data = train.filter(pl.col("tdq").dt.year() < min_year + self.min_train_years)
        neg_count = len(filtered_data.filter(pl.col(target) == 0))
        pos_count = len(filtered_data.filter(pl.col(target) == 1))
        return round(neg_count / pos_count, 2)

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


def validate_trade_date(date: dt.datetime) -> bool:
    """
    Validate if the trade date is one of the allowed dates
    (March 1st, June 1st, September 1st, or December 1st).

    Parameters
    ----------
    date : dt.datetime
        Date to validate.

    Returns
    -------
    bool
        True if date is valid, False otherwise.
    """
    allowed_months = [3, 6, 9, 12]
    return date.month in allowed_months and date.day == 1


def find_last_trading_date() -> Optional[dt.datetime]:
    """
    Find last trading date, which will be used for stock selection.

    Returns
    -------
    dt.datime
        Trading date.
    """

    today = dt.datetime.today()
    trade_dates = [
        dt.datetime(today.year - 1, 12, 1),
        dt.datetime(today.year, 3, 1),
        dt.datetime(today.year, 6, 1),
        dt.datetime(today.year, 9, 1),
        dt.datetime(today.year, 12, 1),
    ]
    past_dates = [date for date in trade_dates if date <= today]

    if past_dates:
        return max(past_dates)
    else:
        logger.error("no trade dates found.")
        return None


def format_ga_parameters(ga_solution: List[float], scale: float) -> dict:
    """
    Format model parameters.

    Parameters
    ----------
    ga_solution : List[float]
        GA solution encoded as a list.
    scale : float
        Class imbalance scale.
    """
    return {
        "objective": "binary:logistic",
        "learning_rate": ga_solution[0],
        "n_estimators": round(ga_solution[1]),
        "max_depth": round(ga_solution[2]),
        "min_child_weight": ga_solution[3],
        "gamma": ga_solution[4],
        "subsample": ga_solution[5],
        "colsample_bytree": ga_solution[6],
        "reg_alpha": ga_solution[7],
        "reg_lambda": ga_solution[8],
        "scale_pos_weight": scale,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "nthread": -1,
        "random_state": 100,
    }
