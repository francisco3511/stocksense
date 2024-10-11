import pygad
import polars as pl
import datetime as dt
from loguru import logger

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
        init_range_high,
        keep_elitism
    ):
        self.num_generations = num_generations
        self.num_parents_mating = num_parents_mating
        self.sol_per_pop = sol_per_pop
        self.num_genes = num_genes
        self.fitness_func = fitness_func
        self.init_range_low = init_range_low
        self.init_range_high = init_range_high
        self.keep_elitism = keep_elitism
        self.random_seed = get_config("model")["seed"]
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
            keep_elitism=self.keep_elitism,
            mutation_percent_genes=20,
            mutation_type="random",
            parent_selection_type="tournament",
            on_generation=self.adaptive_mutation_callback
        )

    def adaptive_mutation_callback(self, ga_instance):
        # adaptive mutation - decrease the mutation rate as the generations increase
        current_generation = ga_instance.generations_completed
        max_generations = ga_instance.num_generations

        # reduce the mutation rate linearly from 10% to 1%
        mutation_decay = 10 - (9 * (current_generation / max_generations))
        ga_instance.mutation_percent_genes = max(1, mutation_decay)

        logger.info(
            f" {current_generation}: Mutation rate set to {ga_instance.mutation_percent_genes}%"
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


def walk_forward_val_split(data, start_year, train_window, val_window):
    """
    Split the dataset into walk-forward training, validation, and test sets.
    """
    current_year = start_year
    while current_year + train_window + val_window < dt.datetime.now().year:
        train = data.filter(
            (pl.col('tdq').dt.year() >= current_year) &
            (pl.col('tdq').dt.year() < current_year + train_window)
        )
        val = data.filter(
            (pl.col('tdq').dt.year() >= current_year + train_window) &
            (pl.col('tdq').dt.year() < current_year + train_window + val_window)
        )
        yield train, val
        current_year += 1


def fitness_function_wrapper(data, date_col, target_col, start_year, train_window, val_window):
    def fitness_function(ga_instance, solution, solution_idx):
        # hyperparameters from GA solution
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

        # init XGBoost model with GA solution parameters
        model = XGBoostModel(params)

        # walk-forward evaluation loop
        perfs = []
        for train, val in walk_forward_val_split(
            data,
            start_year,
            train_window,
            val_window
        ):
            X_train = train.select(pl.exclude(date_col)).to_pandas()
            y_train = train.select(target_col).to_pandas().values.ravel()
            X_val = val.select(pl.exclude(date_col)).to_pandas()
            y_val = val.select(target_col).to_pandas().values.ravel()

            # train on training set
            model.train(X_train, y_train)

            # evaluate performance on validation set
            perf = model.evaluate(X_val, y_val)["pr_auc"]
            perfs.append(perf)

        # average performance across all splits
        avg_perf = sum(perfs) / len(perfs)
        return avg_perf, solution_idx, ga_instance

    return fitness_function
