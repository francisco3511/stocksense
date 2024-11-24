import polars as pl
import pygad
from loguru import logger

from .xgboost_model import XGBoostModel


class GeneticAlgorithm:
    def __init__(self, ga_settings, fitness_func):
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
            parallel_processing=["thread", 4],
        )

    def on_generation(self, ga_instance):
        """
        Callback function.
        """

        best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution()
        logger.info(f"generation {ga_instance.generations_completed}:")
        logger.info(f"\tbest solution: {best_solution}")
        logger.info(f"\tbest fitness: {best_solution_fitness}")

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


def get_train_val_splits(data: pl.DataFrame, stocks: list[str], min_train_years: int = 5):
    """
    Generate training/validation splits using expanding window approach.

    Parameters
    ----------
    data : pl.DataFrame
        Training data to split.
    stocks : list[str]
        List of S&P 500 stocks to include in validation.
    min_train_years : int
        Minimum number of years required for training.

    Returns
    -------
    list[tuple[pl.DataFrame]]
        List of (train, validation) splits.
    """
    # Get unique years in the dataset
    years = data.select(pl.col("tdq").dt.year()).unique().sort("tdq").get_column("tdq").to_list()

    # Ensure we have enough years for training and 2 years of validation
    if len(years) < min_train_years + 2:
        raise ValueError(
            f"Not enough years in dataset. Need at least {min_train_years + 2} years "
            f"({min_train_years} for training, 2 for validation)."
        )

    splits = []
    for i in range(len(years) - 3):
        if i + 1 < min_train_years:
            continue

        train_years = years[: i + 1]
        val_years = [years[i + 2], years[i + 3]]

        train = data.filter(pl.col("tdq").dt.year().is_in(train_years))
        val = data.filter(pl.col("tdq").dt.year().is_in(val_years) & pl.col("tic").is_in(stocks))

        splits.append((train, val))

    return splits


def fitness_function_wrapper(data, features, target, min_train_years, scale, evaluation_stocks):
    splits = get_train_val_splits(data, evaluation_stocks, min_train_years)

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
            "nthread": 1,
            "seed": 100,
        }

        model = XGBoostModel(params)
        perfs = []

        for train, val in splits:
            X_train = train.select(features).to_pandas()
            y_train = train.select(target).to_pandas().values.ravel()
            X_val = val.select(features).to_pandas()
            y_val = val.select(target).to_pandas().values.ravel()

            model.train(X_train, y_train)
            perf = model.pr_auc(X_val, y_val)
            perfs.append(perf)

        return sum(perfs) / len(perfs)

    return fitness_function
