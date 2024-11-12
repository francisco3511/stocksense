import datetime as dt

import polars as pl
import pygad
from config import get_config
from loguru import logger
from model import XGBoostModel


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
        gene_space,
    ):
        self.num_generations = num_generations
        self.num_parents_mating = num_parents_mating
        self.sol_per_pop = sol_per_pop
        self.num_genes = num_genes
        self.fitness_func = fitness_func
        self.init_range_low = init_range_low
        self.init_range_high = init_range_high
        self.gene_space = gene_space
        self.ga_instance = None
        self.best_fitness_value = 0
        self.no_improv_count = 0
        self.no_improv_limit = 5
        self.random_seed = get_config("model")["seed"]

    def create_instance(self):
        logger.info("creating GA instance")
        self.ga_instance = pygad.GA(
            num_generations=self.num_generations,
            num_parents_mating=self.num_parents_mating,
            fitness_func=self.fitness_func,
            sol_per_pop=self.sol_per_pop,
            num_genes=self.num_genes,
            init_range_low=self.init_range_low,
            init_range_high=self.init_range_high,
            gene_space=self.gene_space,
            mutation_percent_genes=10,
            crossover_probability=0.8,
            parent_selection_type="tournament",
            keep_parents=2,
            mutation_type="random",
            crossover_type="single_point",
            on_generation=self.on_generation,
            parallel_processing=-1,
        )

    def on_generation(self, ga_instance):
        """
        Callback function.
        """

        best_solution, best_solution_fitness, best_solution_idx = (
            ga_instance.best_solution()
        )
        logger.info(f"generation {ga_instance.generations_completed}:")
        logger.info(f"\tbest solution: {best_solution}")
        logger.info(f"\tbest fitness: {best_solution_fitness}")

        if best_solution_fitness > self.best_fitness_value:
            self.best_fitness_value = best_solution_fitness
            self.no_improv_count = 0
        else:
            self.no_improv_count += 1

        if self.no_improv_count >= self.no_improv_limit:
            print(
                f"no improvement for {self.no_improv_limit} generations, stopping GA."
            )
            ga_instance.terminate()

    def train(self):
        if self.ga_instance is None:
            raise Exception(
                "GA instance is not created. Call create_instance() before training."
            )
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


def get_train_val_split(
    data: pl.DataFrame, start_year: int, train_window: int, val_window: int
):
    """
    Split the dataset into training and validation sets,
    based on walk-forward validation strategy.

    Parameters
    ----------
    data : pl.DataFrame
        Training data to split.
    start_year : int
        Starting year of walk-forward split.
    train_window : int
        Walk-forward window size.
    val_window : int
        Evaluation window size.

    Returns
    -------
    tuple[pl.DataFrame]
        Training and validation data.
    """

    train = data.filter(
        (pl.col("tdq").dt.year() >= start_year)
        & (pl.col("tdq").dt.year() < start_year + train_window)
    )
    val = data.filter(
        (pl.col("tdq").dt.year() > start_year + train_window)
        & (pl.col("tdq").dt.year() <= start_year + train_window + val_window)
    )
    return train, val


def fitness_function_wrapper(
    data, tic_col, date_col, target_col, start_year, train_window, val_window, scale
):
    def fitness_function(ga_instance, solution, solution_idx):
        params = {
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
            "seed": get_config("model")["seed"],
        }

        model = XGBoostModel(params)
        perfs = []
        window = train_window
        while start_year + window + val_window < dt.datetime.now().year - 1:
            train, val = get_train_val_split(data, start_year, window, val_window)
            X_train = train.select(
                pl.exclude([tic_col, target_col, date_col])
            ).to_pandas()
            y_train = train.select(target_col).to_pandas().values.ravel()
            X_val = val.select(pl.exclude([tic_col, target_col, date_col])).to_pandas()
            y_val = val.select(target_col).to_pandas().values.ravel()

            model.train(X_train, y_train)
            perf = model.evaluate(X_val, y_val)["pr_auc"]
            perfs.append(perf)
            window += 1

        # average performance across all splits
        avg_perf = sum(perfs) / len(perfs)
        return avg_perf

    return fitness_function
