from sklearn.base import RegressorMixin
import pickle
import numpy as np
import gc
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from sklearn.model_selection import KFold, StratifiedKFold
import lightgbm as lgb
import time
from src.helpers import *

import configparser

config = configparser.ConfigParser()
config.read('config.ini')

DATA_PATH = config.get('PATHS', 'DATA_PATH')
MODELS_PATH = config.get('PATHS', 'MODELS_PATH')
features_list = config.get('MODELS', 'features')

with open(os.path.join(DATA_PATH, 'metadata', features_list), 'rb') as f:
    features = pickle.load(f)

default_lgb_params = {'num_leaves': 1023,
                      'max_depth': -1,  # 15
                      'objective': 'regression',
                      'seed': 42,
                      'silent': True,
                      'num_threads': -1,
                      'n_estimators': 1000,
                      'colsample_bytree': 0.3,
                      'subsample': 0.8,
                      'learning_rate': 0.005}


class CustomLGBMRegressor(RegressorMixin):
    """
    Custom Lightgbm Regressor:
    *
    """

    def __init__(self, folds=10,
                 seed=42,
                 save_cv_rounds=True,
                 early_stopping_rounds=300,
                 eval_metric='rmse',
                 best_model_file='LGB3.pkl',
                 **lgb_params):
        """
        Initiate the light gbm regressor parameters
        :param lgb_params:
        """
        self.folds = folds
        self.seed = seed
        self.save_cv_rounds = save_cv_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.lgb_params = default_lgb_params
        self.lgb_params.update(lgb_params)
        self._clf = None
        self.best_model_file = best_model_file

    def fit(self, X=None, y=None):
        """
        Train the model with cross validation technique
        :param X: 
        :param y: 
        :return: 
        """""
        for c in X.columns:
            if X[c].dtype in ['object', 'str']:
                X[c] = X[c].astype('category')
                le = LabelEncoder()
                X[c] = le.fit_transform(X[c].astype(str))

        oof = pd.DataFrame(np.zeros(X.shape[0]), columns=["pred"])
        print(oof.shape)
        CV_SPLITTER = pd.qcut(y, self.folds, labels=False)
        skf = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=self.seed)
        for fold, (idxT, idxV) in enumerate(skf.split(X, CV_SPLITTER)):
            print("FOLD", fold)
            X_train, y_train = X.loc[idxT, features], y[idxT]
            X_test, y_test = X.loc[idxV, features], y[idxV]

            fit_params = {"early_stopping_rounds": self.early_stopping_rounds,
                          'eval_metric': self.eval_metric,
                          "eval_set": [(X_test, y_test)],
                          'eval_names': ['valid'],
                          'feature_name': 'auto',  # that's actually the default
                          'categorical_feature': 'auto'  # that's actually the default
                          }
            clf = lgb.LGBMRegressor(**self.lgb_params)
            START = time.time()
            clf.fit(X_train, y_train, **fit_params)
            END = time.time()
            print("time to train ", round(END - START, 1), "secondes")

            if self.save_cv_rounds:
                joblib.dump(self._clf, os.path.join(f'drug_price_lgb{fold}.pkl'))
            #     clf = joblib.load(f'lgb{fold}.pkl')
            oof.loc[idxV, "pred"] = self._clf.predict(X_test[features])

        self._clf = clf
        self.cv_ = oof
        print("done")
        return self

    def transform(self, X):
        return X

    def predict(self, X):
        if not self._clf:
            self.load_model()
        X = pandas_converter(X)
        return self._clf.predict(X)

    def load_model(self, filename=None):
        if not filename:
            filename = self.best_model_file
        self._clf = joblib.load(os.path.join(MODELS_PATH, filename))

