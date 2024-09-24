import pandas as pd
import numpy as np


class FixFeatures():
    """Class to fix variable's type and missing values
    """

    def __init__(self, cast_features):

        self.cast_features = cast_features

        self.type_function = {
            'float32' : np.float32,
            'float64' : np.float64,
            'str' : str,
            'int64' : np.int64,
            'int32' : np.int32,
            'datetime[64]' : 'datetime64[s]'
        }

    def fit(self, X, y=None):
        return self

    def fix_type(self, X):
        for type, columns in self.cast_features.items():
            X[columns] = X[columns].astype(self.type_function[type])
        return X
    
    def fix_missing(self, X):
        return X

    def transform(self, X):
        X_tmp = X.reset_index(drop=True)
        X_tmp = self.fix_missing(X_tmp)
        X_tmp = self.fix_type(X_tmp)
        return X_tmp