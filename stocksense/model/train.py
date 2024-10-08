import re
import polars as pl

from model import XGBoostModel, GeneticAlgorithm, fitness_function_wrapper


def extract_date(file_name):
    match = re.search(r'\d{4}-\d{2}-\d{2}', file_name)
    return match.group(0) if match else None


def walk_forward_split(data, date_col, start_date, train_window, val_window, test_window):
    """
    Split the dataset into walk-forward training, validation, and test sets.
    """
    current_date = start_date
    while True:
        train = data.filter(
            pl.col(date_col).is_between(
                current_date,
                current_date + train_window
            )
        )
        val = data.filter(
            pl.col(date_col).is_between(
                current_date + train_window,
                current_date + train_window + val_window
            )
        )
        yield train, val
        current_date += test_window


def train():
    """
    Train sector-specific models.
    """

    # Load your dataset in Polars
    data = pl.read_csv("your_financial_data.csv")

    # Parameters for the walk-forward split
    date_col = "report_date"
    start_date = "2015-01-01"
    train_window = pl.duration("P1Y")
    val_window = pl.duration("P3M")
    test_window = pl.duration("P3M")

    # Create GA instance
    ga = GeneticAlgorithm(
        num_generations=50,
        num_parents_mating=5,
        sol_per_pop=10,
        num_genes=9,
        fitness_func=fitness_function_wrapper(
            data,
            date_col,
            start_date,
            train_window,
            val_window,
            test_window
        ),
        init_range_low=[0.01, 100, 3, 1, 0, 0.5, 0.5, 0, 1],
        init_range_high=[0.3, 1000, 10, 10, 5, 1, 1, 10, 10]
    )

    # Train GA to optimize hyperparameters
    ga.create_instance()
    ga.train()
