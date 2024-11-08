import warnings
import datetime as dt
import polars as pl
from pathlib import Path
from loguru import logger

from config import get_config
from .xgboost_model import XGBoostModel
from .genetic_algorithm import GeneticAlgorithm, fitness_function_wrapper


warnings.filterwarnings("ignore")


class ModelHandler:
    """
    Stocksense stock selection model handling class.
    Basic handling for model training, evaluation and testing.
    """

    def __init__(self):
        self.tic_col = "tic"
        self.date_col = "tdq"
        self.target_col = "fperf"
        self.train_start = 2007
        self.train_window = 12
        self.val_window = 2
        self.model_path = Path("models/")

    def train(self, data: pl.DataFrame):
        """
        Train and optimize GA-XGBoost model.

        Parameters
        ----------
        data : pl.DataFrame
            Preprocessed financial data.

        Raises
        ------
        Exception
            If window size overflows.
        """
        try:
            if (
                self.train_start + self.train_window + self.val_window
                > dt.datetime.now().year - 1
            ):
                raise Exception("Window size overflow")

            trade_date = find_last_trading_date()

            logger.info(f"START training model - {trade_date}")

            self.model_path.mkdir(parents=True, exist_ok=True)

            train_df = data.filter(
                (pl.col("tdq") < trade_date)
                & ~pl.all_horizontal(pl.col(self.target_col).is_null())
            )

            # get imbalance approx scale
            scale = int(
                len(
                    train_df.filter(
                        (pl.col(self.target_col) == 0)
                        & (pl.col("tdq").dt.year() >= self.train_start)
                        & (
                            pl.col("tdq").dt.year()
                            < self.train_start + self.train_window
                        )
                    )
                )
                / len(
                    train_df.filter(
                        (pl.col(self.target_col) == 1)
                        & (pl.col("tdq").dt.year() >= self.train_start)
                        & (
                            pl.col("tdq").dt.year()
                            < self.train_start + self.train_window
                        )
                    )
                )
            )

            aux_cols = ["datadate", "rdq", "sector"] + [
                t for t in get_config("model")["targets"] if t != self.target_col
            ]
            train_df = train_df.select(
                [c for c in train_df.columns if c not in aux_cols]
            )

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
                "seed": get_config("model")["seed"],
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

    def score(self):
        """
        Classify using sector-specific models.
        """

        try:
            data = load_processed_data()
            trade_date = find_last_trading_date()

            logger.info(f"START stocksense eval - {trade_date}")

            test_df = data.filter((pl.col("tdq") == trade_date))
            test_df = test_df.filter(
                ~pl.all_horizontal(pl.col(self.target_col).is_null())
            )

            aux_cols = ["datadate", "rdq", "tic", "sector", "freturn", "adj_freturn"]
            test_df = test_df.select([c for c in test_df.columns if c not in aux_cols])
            test_df = test_df.select(
                pl.exclude([self.target_col, self.date_col])
            ).to_pandas()

            model_path = model_path = (
                Path("models/") / f"xgb_{self.last_trade_date}.pkl"
            )
            model = XGBoostModel().load_model(model_path)
            model.predict_proba(test_df)
        except Exception:
            logger.error("ERROR: no model available.")


def find_most_recent(file_dir: Path, format: str = "csv"):
    """
    Find most recent file in directory.

    Parameters
    ----------
    file_dir : Path
        Directory containing files.
    format : str, optional
        File format, by default "csv".

    Returns
    -------
    Path
        Path to most recent file.

    Raises
    ------
    FileNotFoundError
    """
    files = file_dir.glob(f"*.{format}")
    date_files = [
        (file, dt.datetime.strptime(file.stem.split("_")[-1], "%Y-%m-%d"))
        for file in files
    ]
    if date_files:
        return max(date_files, key=lambda x: x[1])[0]
    else:
        raise FileNotFoundError


def load_processed_data():
    """
    Loads last batch of processed data.

    Returns
    -------
    pl.DataFrame
        Production ready training data.

    Raises
    ------
    FileNotFoundError
        If no file is found.
    """
    try:
        directory_path = Path("data/1_work_data/processed")
        most_recent_file = find_most_recent(directory_path)
        return pl.read_csv(most_recent_file, try_parse_dates=True)
    except FileNotFoundError:
        logger.error("no processed data found.")


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
