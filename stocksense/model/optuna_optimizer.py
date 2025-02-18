from typing import List

import numpy as np
import optuna
import polars as pl
from loguru import logger

from .utils import (
    format_xgboost_params,
    get_dataset_imbalance_scale,
    get_train_val_splits,
    round_params,
)
from .xgboost_model import XGBoostClassifier


class OptunaOptimizer:
    """
    Optuna implementation for hyperparameter optimization.

    Attributes
    ----------
    study : optuna.Study
        The Optuna study instance
    best_params : dict
        Best parameters found during optimization
    best_value : float
        Best objective value achieved during optimization
    """

    def __init__(self, n_trials: int = 600):
        self.n_trials = n_trials
        self.study = None
        self.best_params = None
        self.best_value = None
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def create_study(self):
        """Create a new Optuna study."""
        logger.info("Creating Optuna study")

        n_startup = int(self.n_trials * 0.3)

        sampler = optuna.samplers.TPESampler(
            n_startup_trials=n_startup,
            multivariate=False,
            group=False,
            constant_liar=False,

        )

        pruner = optuna.pruners.PercentilePruner(
            percentile=50.0,
            n_startup_trials=n_startup,
            n_warmup_steps=0,
            interval_steps=1,
            n_min_trials=30
        )

        self.study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner
        )

    def optimize(
        self,
        data: pl.DataFrame,
        features: List[str],
        target: str,
        max_train_years: int = 10
    ) -> dict:
        """
        Optimize hyperparameters using Optuna.

        Parameters
        ----------
        data : pl.DataFrame
            Training data
        features : List[str]
            Features to use for training
        target : str
            Target variable to predict
        max_train_years : int
            Maximum number of years to use for training

        Returns
        -------
        dict
            Best parameters found
        """
        if self.study is None:
            self.create_study()

        splits = get_train_val_splits(data, max_train_years, 3)

        # Log validation set sizes for monitoring
        for i, (_, val) in enumerate(splits):
            trade_dates = val.select("tdq").unique().sort(by="tdq")
            for date in trade_dates.to_series():
                n_obs = len(val.filter(pl.col("tdq") == date))
                logger.info(f"Split {i+1} - {date}: {n_obs} observations")

        def objective(trial: optuna.Trial) -> float:
            """
            Optuna objective function.
            """

            base_scale = get_dataset_imbalance_scale(splits[-1][0], target)

            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.4, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 600, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 7.0),
                'gamma': trial.suggest_float('gamma', 1e-3, 1.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 5.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 5.0, log=True),
                'scale_pos_weight': trial.suggest_float(
                    'scale_pos_weight',
                    1.0,
                    max(round(base_scale * 1.2, 1), 2.0),
                    log=True
                ),
            }

            performance_list = []

            for i, (train, val) in enumerate(splits):
                X_train = train.select(features).to_pandas()
                y_train = train.select(target).to_pandas().values.ravel()
                trade_dates = val.select("tdq").to_pandas().values.ravel()

                formatted_params = format_xgboost_params(params, 100)
                xgb = XGBoostClassifier(formatted_params)
                xgb.train(X_train, y_train)

                for trade_date in np.unique(trade_dates):
                    val_trade_date = val.filter(pl.col("tdq") == trade_date)
                    X_val = val_trade_date.select(features).to_pandas()
                    y_true = val_trade_date.select(target).to_pandas().values.ravel()
                    y_pred_proba = xgb.predict_proba(X_val)

                    perf = evaluate_hit_rate_improvement(y_true, y_pred_proba)
                    performance_list.append(perf)

                accumulated_performance = np.mean(performance_list)
                trial.report(accumulated_performance, i)

                if trial.should_prune():
                    logger.warning(
                        f"Trial {trial.number} pruned after split {i+1} "
                        f"({accumulated_performance:.3f})"
                    )
                    raise optuna.TrialPruned()

            return np.mean(performance_list)

        logger.info(f"Starting optimization for {target} with {self.n_trials} trials")

        callbacks = [
            lambda study, trial: (
                logger.info(
                    f"Trial {trial.number} best value: {study.best_value:.3f} "
                    f"({round_params(study.best_params)})"
                ) if trial.number > 0 and trial.number % 10 == 0 else None
            )
        ]

        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            callbacks=callbacks,
            show_progress_bar=False,
            n_jobs=1
        )

        self.best_params = self.study.best_params
        self.best_value = self.study.best_value

        logger.info(f"Final best performance: {self.best_value:.2f}")
        logger.info(f"Final best parameters: {round_params(self.best_params)}")

        return self.best_params


def evaluate_hit_rate_improvement(y_true: np.array, y_pred: np.array) -> float:
    """
    Evaluate hit rates at multiple top-k thresholds relevant for stock selection.
    """
    thresholds = [0.03, 0.05, 0.1]
    scores = []

    for k in thresholds:
        top_k = int(len(y_true) * k)
        top_indices = np.argsort(y_pred)[-top_k:]
        hit_rate = np.mean(y_true[top_indices])
        baseline = np.mean(y_true)
        relative_improvement = (hit_rate / baseline) if baseline > 0 else 0
        scores.append(relative_improvement)

    return np.mean(scores)
