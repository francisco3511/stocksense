import os
from typing import Callable

import numpy as np
import polars as pl
import pygad
from loguru import logger

from .xgboost_model import XGBoostRegressor

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"


class GeneticAlgorithm:
    def __init__(
        self, ga_settings: dict, fitness_func: Callable[[pygad.GA, list[float], int], float]
    ):
        self.num_generations = ga_settings["num_generations"]
        self.num_parents_mating = ga_settings["num_parents_mating"]
        self.sol_per_pop = ga_settings["sol_per_pop"]
        self.num_genes = ga_settings["num_genes"]
        self.fitness_func = fitness_func
        self.init_range_low = ga_settings["init_range_low"]
        self.init_range_high = ga_settings["init_range_high"]
        self.gene_space = ga_settings["gene_space"]
        self.mutation_percent_genes = ga_settings["mutation_percent_genes"]
        self.crossover_probability = ga_settings["crossover_probability"]
        self.parent_selection_type = ga_settings["parent_selection_type"]
        self.keep_parents = ga_settings["keep_parents"]
        self.mutation_type = ga_settings["mutation_type"]
        self.crossover_type = ga_settings["crossover_type"]
        self.initial_mutation_rate = ga_settings["mutation_percent_genes"]
        self.mutation_percent_genes = self.initial_mutation_rate
        self.ga_instance = None
        self.best_fitness_value = 0
        self.no_improv_count = 0
        self.no_improv_limit = 5

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
            mutation_percent_genes=self.mutation_percent_genes,
            crossover_probability=self.crossover_probability,
            parent_selection_type=self.parent_selection_type,
            keep_parents=self.keep_parents,
            mutation_type=self.mutation_type,
            crossover_type=self.crossover_type,
            on_generation=self.on_generation,
            parallel_processing=["thread", 2],
        )

    def on_generation(self, ga_instance: pygad.GA):
        """
        Callback function.

        Parameters
        ----------
        ga_instance : pygad.GA
            Genetic algorithm instance.
        """

        best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution()
        logger.info(f"GENERATION {ga_instance.generations_completed}:")
        logger.info(f"\tBest solution: [{', '.join(f'{val:.2f}' for val in best_solution)}]")
        logger.info(f"\tBest fitness: {best_solution_fitness}")

        if best_solution_fitness > self.best_fitness_value:
            self.best_fitness_value = best_solution_fitness
            self.no_improv_count = 0
        else:
            self.no_improv_count += 1

        if self.no_improv_count > self.no_improv_limit:
            logger.warning(f"no improvement for {self.no_improv_limit} generations, stopping GA.")
            return "stop"
        elif self.no_improv_count > 2:
            self.mutation_percent_genes = min(50, self.mutation_percent_genes * 1.5)
            logger.warning(f"increasing mutation rate to {self.mutation_percent_genes}")
        else:
            self.mutation_percent_genes = self.initial_mutation_rate

        ga_instance.mutation_percent_genes = self.mutation_percent_genes

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
                "GA instance is not created. " "Call create_instance() before plotting fitness."
            )
        self.ga_instance.plot_fitness()


def top_k_performance(y_true: np.array, y_pred: np.array, k: int = 100) -> float:
    """
    Calculate the performance metrics of top K predictions.

    Parameters
    ----------
    y_true : np.array
        Actual returns
    y_pred : np.array
        Predicted ranking scores
    k : int
        Number of top stocks to consider

    Returns
    -------
    dict
        Dictionary containing performance metrics:
        - returns: Average returns of top K predicted stocks
        - hit_rate: Proportion of positive returns in top K
        - rmse: Root mean squared error of predictions
    """
    top_k_indices = np.argsort(y_pred)[-k:]
    top_k_returns = np.mean(y_true[top_k_indices])
    hit_rate = np.mean(y_true[top_k_indices] > 0)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    returns_norm = (top_k_returns - min(y_true)) / (max(y_true) - min(y_true))
    rmse_norm = rmse / (max(y_true) - min(y_true))

    return (0.4 * returns_norm) + (0.3 * hit_rate) + (-0.3 * rmse_norm)


def top_k_returns(y_true: np.array, y_pred: np.array, k: int = 100) -> float:
    """
    Calculate the average returns of top K predictions.

    Parameters
    ----------
    y_true : np.array
        Actual returns
    y_pred : np.array
        Predicted ranking scores
    k : int
        Number of top stocks to consider

    Returns
    -------
    float
        Average returns of top K predicted stocks
    """
    top_k_indices = np.argsort(y_pred)[-k:]
    return np.mean(y_true[top_k_indices])


def get_train_val_splits(
    data: pl.DataFrame, min_train_years: int = 5, val_years: int = 2, max_splits: int = 2
) -> list[tuple[pl.DataFrame, pl.DataFrame]]:
    """
    Generate training/validation splits using expanding window approach,
    starting from most recent years and moving backwards.

    Parameters
    ----------
    data : pl.DataFrame
        Training data to split.
    min_train_years : int
        Minimum number of years required for training
    max_splits : int
        Maximum number of splits to return

    Returns
    -------
    list[tuple[pl.DataFrame]]
        List of (train, validation) splits, ordered from most recent to oldest.
    """
    quarters = (
        data.select(pl.col("tdq")).unique().sort("tdq", descending=True).get_column("tdq").to_list()
    )

    val_years = 2
    min_train_quarters = min_train_years * 4
    val_window = val_years * 4

    # ensure we have enough years for training and validation
    if len(quarters) < min_train_quarters + val_window:
        raise ValueError(
            f"Not enough years in dataset. Need at least {min_train_years + 2} years "
            f"({min_train_years} for training, 2 for validation)."
        )

    splits = []
    for i in range(0, len(quarters) - min_train_quarters - val_window - 1, val_window):
        val_quarters = quarters[i : (i + val_window)]
        train_quarters = quarters[(i + val_window) :]

        if len(train_quarters) < min_train_quarters:
            break

        train = data.filter(pl.col("tdq").is_in(train_quarters))
        val = data.filter(pl.col("tdq").is_in(val_quarters))
        splits.append((train, val))

    if max_splits and max_splits > 0:
        splits = splits[:max_splits]

    return splits


def fitness_function_wrapper(
    data: pl.DataFrame, features: list[str], target: str, min_train_years: int = 5
) -> Callable[[pygad.GA, list[float], int], float]:
    """
    Wrapper for the fitness function used in the genetic algorithm.

    Parameters
    ----------
    data : pl.DataFrame
        Training data.
    features : list[str]
        Features to use for training.
    target : str
        Target variable to predict.
    min_train_years : int
        Minimum number of years to use for training.

    Returns
    -------
    Callable[[pygad.GA, list[float], int], float]
        Fitness function.
    """
    splits = get_train_val_splits(data, min_train_years)

    def fitness_function(ga_instance, solution, solution_idx) -> float:
        """
        Fitness function for the genetic algorithm.

        Parameters
        ----------
        ga_instance : pygad.GA
            Genetic algorithm instance.
        solution : list[float]
            Solution vector.
        solution_idx : int
            Index of the solution.

        Returns
        -------
        float
            Fitness value.
        """
        params = {
            "objective": "reg:squarederror",
            "learning_rate": solution[0],
            "n_estimators": round(solution[1]),
            "max_depth": round(solution[2]),
            "min_child_weight": solution[3],
            "gamma": solution[4],
            "subsample": solution[5],
            "colsample_bytree": solution[6],
            "reg_alpha": solution[7],
            "reg_lambda": solution[8],
            "eval_metric": "rmse",
            "nthread": 2,
            "seed": 100,
        }

        model = XGBoostRegressor(params)
        performance_list = []

        for train, val in splits:
            X_train = train.select(features).to_pandas()
            y_train = train.select(target).to_pandas().values.ravel()
            X_val = val.select(features).to_pandas()
            y_val = val.select(target).to_pandas().values.ravel()

            model.train(X_train, y_train)
            y_pred = model.predict(X_val)
            performance = top_k_performance(y_val, y_pred, k=200)
            performance_list.append(performance)

        avg_performance = np.mean(performance_list)
        return avg_performance if avg_performance > 0 else 0.0001

    return fitness_function
