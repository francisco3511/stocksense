import datetime as dt
import warnings
from pathlib import Path

import polars as pl
from config import config
from loguru import logger

from .genetic_algorithm import GeneticAlgorithm, fitness_function_wrapper
from .xgboost_model import XGBoostModel

PACKAGE_DIR = Path(__file__).parents[1]
MODEL_PATH = PACKAGE_DIR / "models"

warnings.filterwarnings("ignore")


class ModelHandler:
    """
    Stocksense stock selection model handling class.
    Basic handling for model training, evaluation and testing.
    """

    def __init__(self):
        self.id_col = config.model.id_col
        self.date_col = config.model.date_col
        self.target_col = config.model.target
        self.train_start = config.model.train_start
        self.train_window = config.model.train_window
        self.val_window = config.model.val_window
        self.seed = config.model.seed

    def train(self, data: pl.DataFrame):
        """
        Train and optimize GA-XGBoost model.

        Parameters
        ----------
        data : pl.DataFrame
            Preprocessed financial data.
        """
        try:
            trade_date = find_last_trading_date()
            logger.info(f"START training model - {trade_date}")

            train_df = data.filter(
                (pl.col("tdq") < trade_date) & ~pl.all_horizontal(pl.col(self.target_col).is_null())
            )
            scale = self.get_dataset_imbalance_scale(train_df)
            aux_cols = ["datadate", "rdq", "sector"]
            train_df = train_df.select([c for c in train_df.columns if c not in aux_cols])

            ga = GeneticAlgorithm(
                num_generations=50,
                num_parents_mating=10,
                sol_per_pop=50,
                num_genes=9,
                fitness_func=fitness_function_wrapper(
                    train_df,
                    self.tic_col,
                    self.date_col,
                    self.target_col,
                    self.train_start,
                    self.train_window,
                    self.val_window,
                    scale,
                ),
                init_range_low=[0.001, 50, 2, 1, 0, 0.5, 0.5, 0, 0],
                init_range_high=[0.5, 500, 12, 10, 10, 1, 1, 12, 12],
                gene_space=[
                    {"low": 0.001, "high": 0.5},
                    {"low": 50, "high": 500},
                    {"low": 2, "high": 12},
                    {"low": 1, "high": 10},
                    {"low": 0, "high": 10},
                    {"low": 0.5, "high": 1},
                    {"low": 0.5, "high": 1},
                    {"low": 0, "high": 12},
                    {"low": 0, "high": 12},
                ],
            )

            ga.create_instance()
            ga.train()
            best_solution, best_solution_fitness, best_solution_idx = ga.best_solution()

            params = {
                "objective": "binary:logistic",
                "learning_rate": best_solution[0],
                "n_estimators": int(best_solution[1]),
                "max_depth": int(best_solution[2]),
                "min_child_weight": best_solution[3],
                "gamma": best_solution[4],
                "subsample": best_solution[5],
                "colsample_bytree": best_solution[6],
                "reg_alpha": best_solution[7],
                "reg_lambda": best_solution[8],
                "scale_pos_weight": scale,
                "eval_metric": "logloss",
                "nthread": -1,
                "seed": self.seed,
            }

            X_train = train_df.select(
                pl.exclude([self.tic_col, self.target_col, self.date_col])
            ).to_pandas()
            y_train = train_df.select(self.target_col).to_pandas().values.ravel()

            model = XGBoostModel(params, scale=scale)
            model.train(X_train, y_train)
            model.save_model(self.model_path / f"xgb_{trade_date}.pkl")
        except Exception:
            logger.error("ERROR: failed to train model.")

    def get_dataset_imbalance_scale(self, train_df):
        """
        Compute dataset class imbalance scale.

        Parameters
        ----------
        train_df : pl.DataFrame
            Training dataset.

        Returns
        -------
        int
            Class imbalance scale.
        """
        return int(
            len(
                train_df.filter(
                    (pl.col(self.target_col) == 0)
                    & (pl.col("tdq").dt.year() >= self.train_start)
                    & (pl.col("tdq").dt.year() < self.train_start + self.train_window)
                )
            )
            / len(
                train_df.filter(
                    (pl.col(self.target_col) == 1)
                    & (pl.col("tdq").dt.year() >= self.train_start)
                    & (pl.col("tdq").dt.year() < self.train_start + self.train_window)
                )
            )
        )

    def score(self, data):
        """
        Classify using sector-specific models.
        """

        try:
            trade_date = find_last_trading_date()
            logger.info(f"START stocksense eval - {trade_date}")

            test_df = data.filter((pl.col("tdq") == trade_date))
            test_df = test_df.filter(~pl.all_horizontal(pl.col(self.target_col).is_null()))

            aux_cols = ["datadate", "rdq", "tic", "sector", "freturn", "adj_freturn"]
            test_df = test_df.select([c for c in test_df.columns if c not in aux_cols])
            test_df = test_df.select(pl.exclude([self.target_col, self.date_col])).to_pandas()

            model_path = model_path = Path("models/") / f"xgb_{self.last_trade_date}.pkl"
            model = XGBoostModel().load_model(model_path)
            model.predict_proba(test_df)
        except Exception:
            logger.error("ERROR: no model available.")


def find_last_trading_date():
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
