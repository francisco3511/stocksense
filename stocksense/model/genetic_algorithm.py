import pygad
import polars as pl

from model import XGBoostModel
from config import get_config


class GeneticAlgorithm:
    def __init__(
        self,
        num_generations,
        num_parents_mating,
        sol_per_pop,
        num_genes,
        fitness_func,
        init_range_low,
        init_range_high
    ):
        self.num_generations = num_generations
        self.num_parents_mating = num_parents_mating
        self.sol_per_pop = sol_per_pop
        self.num_genes = num_genes
        self.fitness_func = fitness_func
        self.init_range_low = init_range_low
        self.init_range_high = init_range_high
        self.ga_instance = None

    def create_instance(self):
        self.ga_instance = pygad.GA(
            num_generations=self.num_generations,
            num_parents_mating=self.num_parents_mating,
            fitness_func=self.fitness_func,
            sol_per_pop=self.sol_per_pop,
            num_genes=self.num_genes,
            init_range_low=self.init_range_low,
            init_range_high=self.init_range_high,
            mutation_percent_genes=10,
            mutation_type="random"
        )

    def train(self):
        if self.ga_instance is None:
            raise Exception("GA instance is not created. Call create_instance() before training.")
        self.ga_instance.run()

    def best_solution(self):
        if self.ga_instance is None:
            raise Exception(
                "GA instance is not created. "
                "Call create_instance() before retrieving the best solution."
            )
        return self.ga_instance.best_solution()

    def plot_fitness(self):
        if self.ga_instance is None:
            raise Exception(
                "GA instance is not created. "
                "Call create_instance() before plotting fitness."
            )
        self.ga_instance.plot_fitness()


def walk_forward_val_split(data, start_date, train_window, val_window, test_window):
    """
    Split the dataset into walk-forward training, validation, and test sets.
    """
    current_date = start_date
    while True:
        train = data.filter(
            pl.col('tdq').is_between(
                current_date,
                current_date + train_window
            )
        )
        val = data.filter(
            pl.col('tdq').is_between(
                current_date + train_window,
                current_date + train_window + val_window
            )
        )
        yield train, val
        current_date += test_window


def fitness_function_wrapper(data, date_col, start_date, train_window, val_window, test_window):
    def fitness_function(solution, _):
        # Hyperparameters from GA solution
        params = {
            'objective': 'binary:logistic',
            'learning_rate': solution[0],
            'n_estimators': int(solution[1]),
            'max_depth': int(solution[2]),
            'min_child_weight': solution[3],
            'gamma': solution[4],
            'subsample': solution[5],
            'colsample_bytree': solution[6],
            'reg_alpha': solution[7],
            'reg_lambda': solution[8],
            'scale_pos_weight': 1,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'nthread': -1,
            'seed': get_config('model')['seed']
        }

        # Initialize XGBoost model with GA solution parameters
        model = XGBoostModel(params)

        # Walk-forward evaluation loop
        precisions = []
        for train, val, _ in walk_forward_val_split(data, start_date, train_window, val_window):
            X_train = train.select(pl.exclude(date_col)).to_pandas()  # Convert Polars to pandas
            y_train = train.select("target").to_pandas().values.ravel()
            X_val = val.select(pl.exclude(date_col)).to_pandas()
            y_val = val.select("target").to_pandas().values.ravel()

            # Train on training set
            model.train(X_train, y_train)

            # Evaluate precision on validation set
            precision = model.evaluate(X_val, y_val)["prec"]
            precisions.append(precision)

        # Average precision across all splits
        avg_precision = sum(precisions) / len(precisions)
        return avg_precision

    return fitness_function
