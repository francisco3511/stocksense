import pygad

from .xgboost_model import XGBoostModel

class GeneticAlgorithm:
    def __init__(self, num_generations, num_parents_mating, sol_per_pop, num_genes, fitness_func, init_range_low, init_range_high):
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
            raise Exception("GA instance is not created. Call create_instance() before retrieving the best solution.")
        return self.ga_instance.best_solution()

    def plot_fitness(self):
        if self.ga_instance is None:
            raise Exception("GA instance is not created. Call create_instance() before plotting fitness.")
        self.ga_instance.plot_fitness()


def fitness_function(solution, x_train, x_test, y_train, y_test):
    """
    Fitness function to evaluate XGBoost model parameters.
    """
    
    params = {
        'objective': 'binary:logistic',
        'learning_rate': solution[0],
        'n_estimators': solution[1],
        'max_depth': solution[2],
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
        'seed': 123
    }
    
    model = XGBoostModel(params)
    model.train(x_train, y_train)
    precision = model.evaluate(x_test, y_test)["prec"]
    return precision