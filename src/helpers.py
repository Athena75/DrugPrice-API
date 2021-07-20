import pandas as pd


def pandas_converter(X):
    if type(X) == dict:
        X = pd.Series(X).to_frame().T
    if type(X) == pd.core.series.Series:
        X = X.to_frame().T
    return X
