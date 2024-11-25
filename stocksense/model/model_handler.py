import datetime as dt
import warnings
from pathlib import Path
from typing import List, Optional

import polars as pl
from loguru import logger

from stocksense.config import config
from stocksense.database_handler import DatabaseHandler

from .genetic_algorithm import GeneticAlgorithm, fitness_function_wrapper
from .xgboost_model import XGBoostModel

MODEL_DIR = Path(__file__).parents[1] / "model" / "model_base"
REPORT_DIR = Path(__file__).parents[2] / "reports"

warnings.filterwarnings("ignore")


class ModelHandler:
    """
    Stocksense stock selection model handling class.
    Basic handling for model training, evaluation and testing.
    """

    def __init__(self, evaluation_stocks, trade_date: Optional[dt.datetime] = None):
        self.features = config.model.features
        self.target = config.model.target
        self.min_train_years = config.model.min_train_years
        self.evaluation_stocks = evaluation_stocks
        self.trade_date = trade_date if trade_date else find_last_trading_date()

    def train(self, data: pl.DataFrame, retrain: bool = False):
        """
        Train and optimize GA-XGBoost model.

        Parameters
        ----------
        data : pl.DataFrame
            Preprocessed financial data.
        retrain : bool
            Whether to retrain the model for given trade date.
        """
        try:
            logger.info(f"START training model - {self.trade_date}")

            model_file = MODEL_DIR / f"{self.trade_date.date()}.pkl"
            if model_file.exists() and not retrain:
                logger.warning(f"Model already exists for {self.trade_date} - skipping training.")
                return

            train = data.filter(
                (pl.col("tdq") < self.trade_date)
                & ~pl.all_horizontal(pl.col(self.target).is_null())
            )

            id_cols = ["tdq", "tic"]
            training_fields = id_cols + self.features + [self.target]
            train = train.select(training_fields)
            scale = self.get_dataset_imbalance_scale(train)

            # run GA optimization
            ga = GeneticAlgorithm(
                ga_settings=config.model.ga,
                fitness_func=fitness_function_wrapper(
                    train,
                    self.features,
                    self.target,
                    self.min_train_years,
                    scale,
                    self.evaluation_stocks,
                ),
            )

            # run XGB-GA optimization
            ga.create_instance()
            ga.train()
            best_solution, best_solution_fitness, best_solution_idx = ga.best_solution()

            # train final model with best params
            params = format_parameters(best_solution, scale)

            X_train = train.select(self.features).to_pandas()
            y_train = train.select(self.target).to_pandas().values.ravel()

            model = XGBoostModel(params)
            model.train(X_train, y_train)
            model.save_model(model_file)
        except Exception as e:
            logger.error(f"ERROR: failed to train model - {e}")

    def score(self, data):
        """
        Classify using sector-specific models.
        """
        try:
            logger.info(f"START stocksense eval - {self.trade_date}")

            model_file = MODEL_DIR / f"{self.trade_date.date()}.pkl"
            if not model_file.exists():
                raise FileNotFoundError(f"No model found for trade date {self.trade_date}")

            test = data.filter(
                (pl.col("tdq") == self.trade_date) & pl.col("tic").is_in(self.evaluation_stocks)
            )
            test_df = test.select(self.features).to_pandas()

            model = XGBoostModel()
            model.load_model(model_file)
            prob_scores = model.predict_proba(test_df)
            test = test.with_columns(pl.Series("score", prob_scores))
            self.save_scoring_report(test)
            return
        except Exception:
            logger.error("ERROR: no model available.")
            raise

    def save_scoring_report(self, test_data: pl.DataFrame):
        """
        Save scoring report csv.

        Parameters
        ----------
        test_data : pl.DataFrame
            _description_
        """
        try:
            logger.info("START saving scoring report")
            report = test_data.select(["tic", "score", "freturn", "adj_freturn"]).sort(
                "score", descending=True
            )
            report_file = REPORT_DIR / f"report_{self.trade_date.date()}.csv"
            report.write_csv(report_file)
            logger.success(f"END saved scoring report to {report_file}")
        except Exception as e:
            logger.error(f"ERROR failed to save scoring report - {e}")
            raise

    def get_dataset_imbalance_scale(self, train: pl.DataFrame):
        """
        Compute dataset class imbalance scale.

        Parameters
        ----------
        train : pl.DataFrame
            Training dataset.

        Returns
        -------
        float
            Class imbalance scale.
        """
        min_year = pl.col("tdq").dt.year().min()
        filtered_data = train.filter(pl.col("tdq").dt.year() < min_year + self.min_train_years)
        neg_count = len(filtered_data.filter(pl.col(self.target) == 0))
        pos_count = len(filtered_data.filter(pl.col(self.target) == 1))
        return round(neg_count / pos_count, 2)


def get_sp500_stocks():
    return DatabaseHandler().fetch_stock().filter(pl.col("spx_status") == 1)["tic"].to_list()


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


def format_parameters(solution: List[float], scale: float):
    """
    Format model parameters.
    """
    # train final model with best params
    return {
        "objective": "binary:logistic",
        "learning_rate": solution[0],
        "n_estimators": int(solution[1]),
        "max_depth": int(solution[2]),
        "min_child_weight": solution[3],
        "gamma": solution[4],
        "subsample": solution[5],
        "colsample_bytree": solution[6],
        "reg_alpha": solution[7],
        "reg_lambda": solution[8],
        "scale_pos_weight": scale,
        "eval_metric": "logloss",
        "nthread": -1,
        "seed": 100,
    }
