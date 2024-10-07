import pandas as pd
import numpy as np


def get_features_type(X, n_categories, features_to_remove):
    features = {'categorical': [], 'numerical': []}

    for column in set(X.columns) - set(features_to_remove):
        unique = X[column].nunique()

        if unique <= n_categories and X[column].dtype in ['object', 'string', 'categorical']:
            features['categorical'].append(column)
        else:
            features['numerical'].append(column)

    return features['categorical'], features['numerical']

def apply_encoders(X, encoders):

    for encoder in encoders:
        X = encoder.transform(X)

    return X