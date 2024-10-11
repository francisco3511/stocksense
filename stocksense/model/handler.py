import warnings
import datetime as dt
import polars as pl
import pandas as pd
from pathlib import Path

from model import XGBoostModel, GeneticAlgorithm, fitness_function_wrapper


warnings.filterwarnings("ignore")


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


def walk_forward_val_split(data, start_year, train_window, val_window):
    """
    Split the dataset into walk-forward training, validation, and test sets using Pandas.
    """
    current_year = start_year
    while current_year + train_window + val_window < dt.datetime.now().year:
        # filter training set
        train = data[
            (data.dt.year >= current_year) &
            (data['tdq'].dt.year < current_year + train_window)
        ]

        # filter validation set
        val = data[
            (data['tdq'].dt.year >= current_year + train_window) &
            (data['tdq'].dt.year < current_year + train_window + val_window)
        ]

        return train, val
        current_year += 1


def train():
    """
    Train sector-specific models.
    """

    # load training data
    data = load_processed_data()

    for sector in data['sector'].unique():
        # sector folder
        sector_path = Path(f"models/{sector.lower()}")
        sector_path.mkdir(parents=True, exist_ok=True)

        # slice sector data and convert to pd df
        df = data.filter(
            (pl.col('tdq').dt.year() >= 2007) &
            (pl.col('tdq').dt.year() < dt.datetime.now().year) &
            (pl.col('sector') == sector)
        ).to_pandas().set_index(['tdq'])

        # filter cols
        aux_cols = ['datadate', 'rdq', 'tdq', 'tic', 'sector', 'freturn', 'adj_freturn']
        df = df[[c for c in df.columns if c not in aux_cols]]


        data = walk_forward_val_split(df, 2007, 12, 2)

        # create GA instance
        ga = GeneticAlgorithm(
            num_generations=50,
            num_parents_mating=20,
            sol_per_pop=50,
            num_genes=9,
            fitness_func=fitness_function_wrapper(df, 'tdq', 'adj_fperf', 2007, 12, 2),
            init_range_low=[0.01, 100, 3, 1, 0, 0.5, 0.5, 0, 1],
            init_range_high=[0.3, 1000, 10, 10, 5, 1, 1, 10, 10],
            keep_elitism=5
        )

        # train GA to optimize hyperparameters
        ga.create_instance()
        ga.train()
        best_solution, best_solution_fitness, best_solution_idx = ga.best_solution()

        model = XGBoostModel(best_solution)
        #  model.train(X_train, y_train)


def load():
    pass
