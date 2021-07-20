import pandas as pd
from pandas import Timestamp
from sklearn.pipeline import Pipeline
from src.custom_transformer import LGBTransformer
from src.model import CustomLGBMRegressor
import os
import pickle
from src.helpers import *


pipe = Pipeline([
    ("km", LGBTransformer()),
    ("lr", CustomLGBMRegressor())
])

def get_shap_explain(X, features, explainer, top=10):
 X = pandas_converter(X)
 shap_sample = explainer.shap_values(X.loc[:, features])
 result = dict(zip(features, shap_sample[0]))
 keys = list(result.keys())
 agg = {}
 iterator = {}
 for k in keys:
  if 'svd' in k:
   v = result.pop(k)
   key = k.split('_svd')[0]
   agg[key] = agg.get(key, 0.0) + v
   iterator[key] = iterator.get(key, 0) + 1
 for k in agg:
  result[k] = agg[k]/iterator[k]
 return dict(sorted(result.items(), key=lambda i: -abs(i[1]))[:top])




