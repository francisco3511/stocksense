import warnings
import datetime as dt
import polars as pl
from pathlib import Path
from loguru import logger

from model import XGBoostModel, GeneticAlgorithm, fitness_function_wrapper


warnings.filterwarnings("ignore")


class ModelHandler:
    """
    Stocksense stock selection model handling class.
    """

    def __init__(self):
        self.status = True
        self.train_start = 2007
        self.date_col = 'tdq'
        self.target_col = 'adj_fperf'
        self.val_train_window = 12
        self.val_eval_window = 2
        self.last_trade_date = None

    def train(self):
        """
        Train sector-specific models.
        """

        logger.info("START training stocksense models")

        # load training data
        data = load_processed_data()

        for sector in data['sector'].unique():
            # log run
            logger.info(f"START training {sector} sector model")

            # sector folder
            sector_path = Path(f"models/{sector.lower()}")
            sector_path.mkdir(parents=True, exist_ok=True)

            # find last trade date
            self.last_trade_date = find_last_trading_date()

            # set training period with quarter gap
            train_df = data.filter(
                (pl.col('tdq') < self.last_trade_date) &
                (pl.col('sector') == sector)
            )
            train_df = train_df.filter(~pl.all_horizontal(pl.col(self.target_col).is_null()))

            # filter cols
            aux_cols = ['datadate', 'rdq', 'tic', 'sector', 'freturn', 'adj_freturn']
            train_df = train_df.select([c for c in train_df.columns if c not in aux_cols])

            # create GA instance
            ga = GeneticAlgorithm(
                num_generations=50,
                num_parents_mating=20,
                sol_per_pop=50,
                num_genes=9,
                fitness_func=fitness_function_wrapper(
                    train_df,
                    self.date_col,
                    self.target_col,
                    self.train_start,
                    self.val_train_window,
                    self.val_eval_window
                ),
                init_range_low=[0.01, 100, 3, 1, 0, 0.5, 0.5, 0, 1],
                init_range_high=[0.3, 1000, 10, 10, 5, 1, 1, 10, 10],
                gene_space=[
                    {'low': 0.01, 'high': 0.3},
                    {'low': 100, 'high': 1000},
                    {'low': 3, 'high': 10},
                    {'low': 1, 'high': 10},
                    {'low': 0, 'high': 5},
                    {'low': 0.5, 'high': 1},
                    {'low': 0.5, 'high': 1},
                    {'low': 0, 'high': 10},
                    {'low': 1, 'high': 10}
                ],
                keep_elitism=5
            )

            # train GA to optimize hyperparameters
            ga.create_instance()
            ga.train()
            best_solution, best_solution_fitness, best_solution_idx = ga.best_solution()

            # train xgboost with best parameter set
            X_train = train_df.select(pl.exclude([self.target_col, self.date_col])).to_pandas()
            y_train = train_df.select(self.target_col).to_pandas().values.ravel()
            model = XGBoostModel(best_solution)
            model.train(X_train, y_train)
            model.save_model(sector_path / f"{sector}_{self.last_trade_date}.pkl")

    def score(self):
        """
        Classify using sector-specific models.
        """

        logger.info("START training stocksense models")

        # load training data
        data = load_processed_data()

        for sector in data['sector'].unique():
            # log run
            logger.info("START scoring {sector} stocks")

            # sector folder
            sector_path = Path(f"models/{sector.lower()}")

            # find last trade date
            self.last_trade_date = find_last_trading_date()

            # set scoring period to last trade date
            test_df = data.filter(
                (pl.col('tdq') == self.last_trade_date) &
                (pl.col('sector') == sector)
            )
            test_df = test_df.filter(~pl.all_horizontal(pl.col(self.target_col).is_null()))

            # filter cols
            aux_cols = ['datadate', 'rdq', 'tic', 'sector', 'freturn', 'adj_freturn']
            test_df = test_df.select([c for c in test_df.columns if c not in aux_cols])
            test_df = test_df.select(
                pl.exclude([self.target_col, self.date_col])
            ).to_pandas()

            try:
                model_path = sector_path / f"{sector}_{self.last_trade_date}.pkl"
                model = XGBoostModel().load_model(model_path)
                model.predict_proba(test_df)
            except Exception:
                logger.error(f"ERROR {sector} {self.last_trade_date }: no model available.")


def load_processed_data():
    """
    Read most recently processed dataset.

    Returns
    -------
    pl.DataFrame
        Production ready training data.

    Raises
    ------
    FileNotFoundError
        If no file is found.
    """

    directory_path = Path("data/1_work_data/processed")
    csv_files = directory_path.glob("*.csv")

    date_files = [
        (file, dt.datetime.strptime(file.stem.split('_')[-1], "%Y-%m-%d"))
        for file in csv_files
    ]

    # find the most recent file
    if date_files:
        most_recent_file = max(date_files, key=lambda x: x[1])[0]
        return pl.read_csv(most_recent_file, try_parse_dates=True)
    else:
        raise FileNotFoundError


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
