import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import sklearn.metrics as skm
import xgboost as xgb


class BaseXGBoostModel:
    """
    Base wrapper for XGBoost models.
    """

    def __init__(self, params: Optional[dict] = None):
        self.params = params if params else self._default_params()
        self.model = None

    def _default_params(self):
        return {
            "learning_rate": 0.1,
            "n_estimators": 100,
            "max_depth": 3,
            "min_child_weight": 1,
            "gamma": 0,
            "subsample": 1,
            "colsample_bytree": 1,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "nthread": 2,
            "seed": 100,
        }

    def save_model(self, model_path: Path) -> None:
        if self.model is None:
            raise Exception("model is not trained yet, train the model before saving.")
        with open(model_path, "wb") as f:
            pickle.dump((self.model, self.params), f)

    def load_model(self, model_path: Path) -> None:
        with open(model_path, "rb") as f:
            self.model, self.params = pickle.load(f)

    def get_importance(self, importance_type: str = "gain") -> list[tuple[str, float]]:
        importance = self.model.get_booster().get_score(importance_type=importance_type)
        return sorted(importance.items(), key=lambda x: x[1], reverse=True)


class XGBoostClassifier(BaseXGBoostModel):
    """
    Wrapper for XGBoost classifier.
    """

    def _default_params(self):
        params = super()._default_params()
        params.update(
            {
                "objective": "binary:logistic",
                "scale_pos_weight": 1.0,
                "eval_metric": "logloss",
            }
        )
        return params

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X_train, y_train, verbose=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise Exception("Model is not trained yet. Train the model before predicting.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise Exception("Model is not trained yet. Train the model before predicting.")
        return self.model.predict_proba(X)[:, 1]

    def get_pr_auc(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        y_proba = self.predict_proba(X_test)
        return skm.average_precision_score(y_test, y_proba)

    def get_roc_auc(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        y_proba = self.predict_proba(X_test)
        return skm.roc_auc_score(y_test, y_proba)

    def get_ndcg_score(
        self, X_test: np.ndarray, y_test: np.ndarray, k: Optional[int] = None
    ) -> float:
        y_proba = self.predict_proba(X_test).reshape(1, -1)
        return skm.ndcg_score(y_test.reshape(1, -1), y_proba, k=k)


class XGBoostRegressor(BaseXGBoostModel):
    """
    Wrapper for XGBoost regressor.
    """

    def _default_params(self):
        params = super()._default_params()
        params.update(
            {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "nthread": 2,
                "seed": 100,
            }
        )
        return params

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X_train, y_train, verbose=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise Exception("Model is not trained yet.")
        return self.model.predict(X)

    # Regression metrics
    def get_mse(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        y_pred = self.predict(X_test)
        return skm.mean_squared_error(y_test, y_pred)

    def get_rmse(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        return np.sqrt(self.get_mse(X_test, y_test))

    def get_mae(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        y_pred = self.predict(X_test)
        return skm.mean_absolute_error(y_test, y_pred)

    def get_r2(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        y_pred = self.predict(X_test)
        return skm.r2_score(y_test, y_pred)
