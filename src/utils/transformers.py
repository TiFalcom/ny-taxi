import pandas as pd
import numpy as np
from datetime import datetime


def get_period_of_day(hour):
    if hour < 6:
        return 'dawn'
    if hour < 12:
        return 'morning'
    if hour < 18:
        return 'afternoon'
    if hour < 24:
        return 'evening'


class FixFeaturesMissing():
    """ Class to fix features's missing values
    """
    def __init__(self, missing_features):
        self.missing_features = missing_features

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        for missing_value, features in self.missing_features.items():
            X.loc[:, features] = X.loc[:, features].fillna(missing_value)
        return X
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return X

class FixFeaturesType():
    """ Class to fix features's type
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
            # TODO: Change to loc
            X[columns] = X[columns].astype(self.type_function[type])
        return X

    def transform(self, X):
        X_tmp = X.reset_index(drop=True)
        X_tmp = self.fix_type(X_tmp)
        return X_tmp
    

class FeatureEngineering():
    """Class to create new features based on existing ones
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def create_payload_features(self, X):
        X = X.reset_index(drop=True)
        X_tmp = X.reset_index(drop=True)

        X_tmp['day_of_week'] = X_tmp['tpep_pickup_datetime'].dt.weekday

        X_tmp['hour_of_day'] = X_tmp['tpep_pickup_datetime'].dt.hour

        X_tmp['period_of_day'] = X_tmp['hour_of_day'].apply(get_period_of_day)

        X_tmp['year_month_day'] = X_tmp['tpep_pickup_datetime'].apply(lambda x:datetime.strftime(x, '%Y%m%d')).astype(int)

        X_tmp['year_month_day_hour'] = X_tmp['tpep_pickup_datetime'].dt.strftime('%Y-%m-%d %H')

        new_columns = [col for col in X_tmp.columns if col not in X.columns]
        X_tmp = X.merge(X_tmp[new_columns], left_index=True, right_index=True, how='left')
        
        return X_tmp

    def transform(self, X):
        return self.create_payload_features(X)
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
class FeatureEngineeringLag():
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def create_lag_aux_features(self, X, interval_col, n=3):
        X = X.reset_index(drop=True)
        X_tmp = X[[interval_col]].reset_index(drop=True)
        for i in range(1,n+1):
            X_tmp[f'{interval_col}_l{i}'] = (pd.to_datetime(X_tmp[interval_col], format='%Y-%m-%d %H') - pd.Timedelta(hours=i)).dt.strftime('%Y-%m-%d %H')
        
        new_columns = [col for col in X_tmp.columns if col not in X.columns]
        X_tmp = X.merge(X_tmp[new_columns], how='left', left_index=True, right_index=True)

        return X_tmp
    
    def merge_with_lag(self, X, interval_col, merge_cols, n=3):
        X = X.reset_index(drop=True)
        X_tmp = X.reset_index(drop=True)

        for i in range(1,n+1):
            X_tmp = (
                X
                .rename(columns={
                    col : col + f'_l{i}'
                    for col in X.columns
                    if col not in [interval_col] + merge_cols and
                        sum([col in col_aux for col_aux in [interval_col + f'_l{j}' for j in range(1,n+1)]]) == False
                })
                .merge(
                    X_tmp,
                    how='right',
                    right_on=[interval_col]+merge_cols,
                    left_on=[interval_col + f'_l{i}'] + merge_cols,
                    suffixes = ('_drop', '')
                )
            )
            X_tmp = X_tmp.drop(columns=[
                col for col in X_tmp.columns if '_drop' in col
            ])

        new_columns = [col for col in X_tmp.columns if col not in X.columns]
        X_tmp = X.merge(X_tmp[new_columns], how='left', left_index=True, right_index=True)
        return X_tmp

    def transform(self, X):
        X = X.reset_index(drop=True)
        X_tmp = X.reset_index(drop=True)

        X_tmp = self.create_lag_aux_features(X_tmp, 'year_month_day_hour', 3)

        X_tmp = self.merge_with_lag(X_tmp, 'year_month_day_hour', ['PULocationID'], 3).fillna(-1)

        new_columns = [col for col in X_tmp.columns if col not in X.columns]
        X_tmp = X.merge(X_tmp[new_columns], how='left', left_index=True, right_index=True)
        return X_tmp
        

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class Selector():
    """ Class to select columns from dataframe
    """
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.loc[:, self.columns]
    
