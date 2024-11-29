import pickle

import sklearn.metrics as skm
import xgboost as xgb


class XGBoostModel:
    """
    Custom wrapper for XGBoost classifier.
    """

    def __init__(self, params=None):
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
                "scale_pos_weight": 1.0,
                "eval_metric": "logloss",
                "nthread": -1,
                "seed": 100,
            }
        )
        self.model = None

    def train(self, X_train, y_train):
        """
        Train the classifier.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training data.
        y_train : pd.Series
            Training labels.
        """
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X_train, y_train, verbose=True)

    def predict(self, X):
        """
        Predict the class labels for the provided data.

        Parameters
        ----------
        X : pd.DataFrame
            Data to predict.

        Returns
        -------
        np.ndarray
            Predicted class labels.

        Raises
        ------
        Exception
            If the model is not trained yet.
        """
        if self.model is None:
            raise Exception("Model is not trained yet. Train the model before predicting.")
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict the class probabilities for the provided data.

        Parameters
        ----------
        X : pd.DataFrame
            Data to predict.

        Returns
        -------
        np.ndarray
            Predicted class probabilities.
        """
        if self.model is None:
            raise Exception("Model is not trained yet. Train the model before predicting.")
        return self.model.predict_proba(X)[:, 1]

    def get_pr_auc(self, X_test, y_test):
        y_proba = self.predict_proba(X_test)
        return skm.average_precision_score(y_test, y_proba)

    def get_roc_auc(self, X_test, y_test):
        y_proba = self.predict_proba(X_test)
        return skm.roc_auc_score(y_test, y_proba)

    def get_ndcg_score(self, X_test, y_test, k=None):
        y_proba = self.predict_proba(X_test).reshape(1, -1)
        return skm.ndcg_score(y_test.reshape(1, -1), y_proba, k=k)

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
