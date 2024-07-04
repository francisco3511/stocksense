import xgboost as xgb
import pickle
from pathlib import Path
import sklearn.metrics as skm


class XGBoostModel:
    def __init__(self, params=None):
        self.params = params if params else {
            'objective': 'binary:logistic',
            'learning_rate': 0.1,
            'n_estimators': 100,
            'max_depth': 3,
            'min_child_weight': 1,
            'gamma': 0,
            'subsample': 1,
            'colsample_bytree': 1,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'scale_pos_weight': 1,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'nthread': 2,
            'seed': 123
        }
        self.model_path = Path('./models')
        self.model = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=True)
        
    def predict(self, X):
        if self.model is None:
            raise Exception("Model is not trained yet. Train the model before predicting.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if self.model is None:
            raise Exception("Model is not trained yet. Train the model before predicting.")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
  
        eval = {
            'acc': skm.accuracy_score(y_test, y_pred),
            'prec': skm.precision_score(y_test, y_pred),
            'f1': skm.f1_score(y_test, y_pred),
            'wf1': skm.f1_score(y_test, y_pred, average='weighted'),
            'rec': skm.recall_score(y_test, y_pred),
            'roc_auc': skm.roc_auc_score(y_test, y_proba),
            'brier': skm.brier_score_loss(y_test, y_proba),
            'pr_auc': skm.average_precision_score(y_test, y_proba),
        }
        
        return eval
    
    def save_model(self, name):
        if self.model is None:
            raise Exception("Model is not trained yet. Train the model before saving.")
        with open(self.model_path / f"{name}.pkl", 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self, name):
        with open(self.model_path / f"{name}.pkl", 'rb') as f:
            self.model = pickle.load(f)
