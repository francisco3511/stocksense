from typing import Callable

import numpy as np
import polars as pl
import pygad
from loguru import logger

from .xgboost_model import XGBoostClassifier


class GeneticAlgorithm:
    """
    Genetic Algorithm implementation for hyperparameter optimization.

    Attributes
    ----------
    ga_instance : Optional[pygad.GA]
        The PyGAD genetic algorithm instance
    best_fitness_value : float
        Best fitness value achieved during training
    """

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
        logger.info("Creating GA instance")
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
            parallel_processing=None,
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

        if self.no_improv_count >= self.no_improv_limit:
            logger.warning(f"No improvement for {self.no_improv_limit} generations, stopping GA.")
            return "stop"
        elif self.no_improv_count >= 2:
            self.mutation_percent_genes = min(50, self.mutation_percent_genes * 1.5)
            logger.warning(f"Increasing mutation rate to {self.mutation_percent_genes}")
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


def evaluate_predictions(
    y_true: np.array, y_pred: np.array, trade_dates: np.array, k: int = 75
) -> float:
    """
    Evaluate predictions by selecting top k stocks for each trade date.

    Parameters:
    -----------
    y_true : np.array
        Actual returns
    y_pred : np.array
        Predicted returns
    trade_dates : np.array
        Array of trade dates for each observation
    k : int
        Number of stocks to select per trade date
    """
    unique_dates = np.unique(trade_dates)
    performance_by_date = []

    for date in unique_dates:
        date_mask = trade_dates == date
        date_true = y_true[date_mask]
        date_pred = y_pred[date_mask]

        top_k_indices = np.argsort(date_pred)[-k:]
        bottom_k_indices = np.argsort(date_pred)[:k]

        top_k_returns = date_true[top_k_indices]
        bottom_k_returns = date_true[bottom_k_indices]

        top_mean_return = np.mean(top_k_returns)
        bottom_mean_return = np.mean(bottom_k_returns)
        return_spread = top_mean_return - bottom_mean_return

        top_hit_rate = np.mean(top_k_returns > 0)
        bottom_hit_rate = np.mean(bottom_k_returns < 0)

        if max(date_true) == min(date_true):
            norm_spread = 0.5
        else:
            norm_spread = (return_spread - min(date_true)) / (max(date_true) - min(date_true))

        performance_by_date.append(
            {
                "date": date,
                "return_spread": norm_spread,
                "top_hit_rate": top_hit_rate,
                "bottom_hit_rate": bottom_hit_rate,
            }
        )

    avg_spread = np.mean([p["return_spread"] for p in performance_by_date])
    avg_top_hit = np.mean([p["top_hit_rate"] for p in performance_by_date])
    avg_bottom_hit = np.mean([p["bottom_hit_rate"] for p in performance_by_date])

    # combined metric of return spread
    performance = 0.5 * avg_spread + 0.25 * avg_top_hit + 0.25 * avg_bottom_hit
    return round(performance, 4) if performance > 0 else 0.0001


def evaluate_top_hit_rate(
    y_true: np.array, y_pred: np.array, trade_dates: np.array, k: int = 75
) -> float:
    """
    Evaluate predictions by selecting top k stocks for each trade date.

    Parameters:
    -----------
    y_true : np.array
        Actual returns
    y_pred : np.array
        Predicted returns
    trade_dates : np.array
        Array of trade dates for each observation
    k : int
        Number of stocks to select per trade date

    Returns
    -------
    float
        Average top hit rate.
    """
    unique_dates = np.unique(trade_dates)
    performance_by_date = []

    for date in unique_dates:
        date_mask = trade_dates == date
        date_true = y_true[date_mask]
        date_pred = y_pred[date_mask]

        top_k_indices = np.argsort(date_pred)[-k:]
        top_hit_rate = np.mean(date_true[top_k_indices] > 0)

        performance_by_date.append(
            {
                "date": date,
                "top_hit_rate": top_hit_rate,
            }
        )

    avg_top_hit = np.mean([p["top_hit_rate"] for p in performance_by_date])
    return round(avg_top_hit, 4) if avg_top_hit > 0 else 0.0001


def get_train_val_splits(
    data: pl.DataFrame, min_train_years: int = 5, val_years: int = 1, max_splits: int = 3
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
    # Get sorted unique quarters
    quarters = (
        data.select(pl.col("tdq")).unique().sort("tdq", descending=True).get_column("tdq").to_list()
    )

    # Convert years to quarters
    min_train_quarters = min_train_years * 4
    val_window = val_years * 4

    # Validate enough data exists
    if len(quarters) < min_train_quarters + val_window:
        raise ValueError(
            f"Not enough years in dataset. Need at least {min_train_years + 2} years "
            f"({min_train_years} for training, 2 for validation)."
        )

    splits = []
    # Generate splits moving backwards through time
    for i in range(0, len(quarters) - min_train_quarters - val_window - 1, val_window):
        # Define validation and training periods (skip 1 quarter for look-ahead bias)
        val_quarters = quarters[i : (i + val_window)]
        train_quarters = quarters[(i + val_window + 4) :]

        if len(train_quarters) < min_train_quarters:
            break

        train = data.filter(pl.col("tdq").is_in(train_quarters))
        val = data.filter(pl.col("tdq").is_in(val_quarters))
        splits.append((train, val))

    if max_splits and max_splits > 0:
        splits = splits[:max_splits]

    return splits[::-1]


def fitness_function_wrapper(
    data: pl.DataFrame, features: list[str], target: str, scale: float, min_train_years: int = 5
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
    splits = get_train_val_splits(data, min_train_years, 1, 2)

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
            "objective": "binary:logistic",
            "learning_rate": solution[0],
            "n_estimators": round(solution[1]),
            "max_depth": round(solution[2]),
            "min_child_weight": solution[3],
            "gamma": solution[4],
            "subsample": solution[5],
            "colsample_bytree": solution[6],
            "reg_alpha": solution[7],
            "reg_lambda": solution[8],
            "scale_pos_weight": scale,
            "eval_metric": "logloss",
            "tree_method": "hist",
            "nthread": -1,
            "random_state": 100,
        }

        xgb = XGBoostClassifier(params)
        performance_list = []

        for train, val in splits:
            X_train = train.select(features).to_pandas()
            y_train = train.select(target).to_pandas().values.ravel()
            X_val = val.select(features).to_pandas()
            y_val = val.select(target).to_pandas().values.ravel()
            trade_dates = val.select("tdq").to_pandas().values.ravel()

            xgb.train(X_train, y_train)
            y_pred = xgb.predict_proba(X_val)

            performance = evaluate_top_hit_rate(y_val, y_pred, trade_dates)
            performance_list.append(performance)

        avg_performance = round(np.mean(performance_list), 4)
        return avg_performance

    return fitness_function
