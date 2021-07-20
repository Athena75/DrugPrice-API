from sklearn.base import BaseEstimator, TransformerMixin
import pickle
import numpy as np
import pandas as pd
import gc
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from src.helpers import *

import configparser

config = configparser.ConfigParser()
config.read('config.ini')

DATA_PATH = config.get('PATHS', 'DATA_PATH')
MODELS_PATH = config.get('PATHS', 'MODELS_PATH')
features_list = config.get('MODELS', 'features')


gc.enable()

with open(os.path.join(DATA_PATH, 'metadata', features_list), 'rb') as f:
    features = pickle.load(f)


class LGBTransformer(TransformerMixin):
    def __init__(self, features=features, le_file='le.joblib'):
        self.features = features
        self.le_ = joblib.load(os.path.join(MODELS_PATH,  le_file))

    def fit(self, X):
        for c in self.features:
            if X[c].dtype in ['object', 'str']:
                X[c] = X[c].astype('category')
                self.le_ = LabelEncoder()
                X[c] = self.le_.fit_transform(X[c].astype(str))
        return self


    def transform(self, X):
        # Need to reshape into a column vector in order to use
        le_dict = dict(zip(self.le_.classes_, self.le_.transform(self.le_.classes_)))
        X = pandas_converter(X)
        X = X.loc[:, features]
        for c in features:
            if X[c].dtype in ['object', 'str']:
                X[c] = X[c].astype('category')
                X[c] = X[c].astype(str).apply(lambda x: le_dict.get(x, -1))
        return X
