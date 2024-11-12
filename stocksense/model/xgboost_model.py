import pickle

import sklearn.metrics as skm
import xgboost as xgb
from config import get_config


class XGBoostModel:
    def __init__(self, params=None, scale=1.0):
        self.params = (
            params
            if params
            else {
                "objective": "binary:logistic",
                "learning_rate": 0.1,
                "n_estimators": 100,
                "max_depth": 3,
                "min_child_weight": 1,
                "gamma": 0,
                "subsample": 1,
                "colsample_bytree": 1,
                "reg_alpha": 0,
                "reg_lambda": 1,
                "scale_pos_weight": scale,
                "eval_metric": "logloss",
                "nthread": -1,
                "seed": get_config("model")["seed"],
            }
        )
        self.model = None

    def train(self, X_train, y_train):
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X_train, y_train, verbose=True)

    def predict(self, X):
        if self.model is None:
            raise Exception(
                "Model is not trained yet. Train the model before predicting."
            )
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model is None:
            raise Exception(
                "Model is not trained yet. Train the model before predicting."
            )
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        y_pred_custom = (y_proba > 0.70).astype(int)

        eval = {
            "acc": skm.accuracy_score(y_test, y_pred),
            "prec": skm.precision_score(y_test, y_pred_custom),
            "f1": skm.f1_score(y_test, y_pred),
            "wf1": skm.f1_score(y_test, y_pred, average="weighted"),
            "rec": skm.recall_score(y_test, y_pred),
            "roc_auc": skm.roc_auc_score(y_test, y_proba),
            "brier": skm.brier_score_loss(y_test, y_proba),
            "pr_auc": skm.average_precision_score(y_test, y_proba),
        }

        return eval

    def get_importance(self, importance_type="gain"):
        importance = self.model.get_booster().get_score(importance_type=importance_type)
        return sorted(importance.items(), key=lambda x: x[1], reverse=True)

    def save_model(self, model_path):
        if self.model is None:
            raise Exception("model is not trained yet, train the model before saving.")
        with open(model_path, "wb") as f:
            pickle.dump((self.model, self.params), f)

    def load_model(self, model_path):
        with open(model_path, "rb") as f:
            self.model, self.params = pickle.load(f)
