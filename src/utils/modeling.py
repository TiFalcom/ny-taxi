import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error


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

def p99(x):
    return x.quantile(0.99)

def p75(x):
    return x.quantile(0.75)

def get_metrics(list_of_tuples_of_models, train_data, valid_data, target_col):
    dict_models_metrics = {
        'model' : ['Default'],
        'desc' : ['Default values mean + std'],
        'rmse_train' : [f'{train_data[target_col].mean():.2f} +- {train_data[target_col].std():.2f}'],
        'rmse_valid' : [f'{valid_data[target_col].mean():.2f} +- {valid_data[target_col].std():.2f}'],
        'diff_rmse' : [0],
        'mae_train' : [f'{train_data[target_col].mean():.2f} +- {train_data[target_col].std():.2f}'],
        'mae_valid' : [f'{valid_data[target_col].mean():.2f} +- {valid_data[target_col].std():.2f}'],
        'diff_mae' : [0]
    }

    for model, features, desc, train_data_, valid_data_ in list_of_tuples_of_models:
        dict_models_metrics['model'].append(model.__class__)
        dict_models_metrics['desc'].append(desc)
        (
            dict_models_metrics['rmse_train'].append(
                root_mean_squared_error(
                    train_data_[target_col],
                    np.nan_to_num(model.predict(train_data_[features]),0)
                )
            )
        )
        (
            dict_models_metrics['rmse_valid'].append(
                root_mean_squared_error(
                    valid_data_[target_col],
                    np.nan_to_num(model.predict(valid_data_[features]),0)
                )
            )
        )
        dict_models_metrics['diff_rmse'].append(
            dict_models_metrics['rmse_train'][-1] - dict_models_metrics['rmse_valid'][-1]
        )

        (
            dict_models_metrics['mae_train'].append(
                mean_absolute_error(
                    train_data_[target_col],
                    np.nan_to_num(model.predict(train_data_[features]),0)
                )
            )
        )
        (
            dict_models_metrics['mae_valid'].append(
                mean_absolute_error(
                    valid_data_[target_col],
                    np.nan_to_num(model.predict(valid_data_[features]),0)
                )
            )
        )
        dict_models_metrics['diff_mae'].append(
            dict_models_metrics['mae_train'][-1] - dict_models_metrics['mae_valid'][-1]
        )


    return pd.DataFrame(dict_models_metrics)


def get_boxplot_stats(data, label):
    stats = {
        "min": np.min(data),
        "q1": np.percentile(data, 25),
        "median": np.median(data),
        "q3": np.percentile(data, 75),
        "max": np.max(data)
    }
    return f"{label} - Min: {stats['min']:.2f}, Q1: {stats['q1']:.2f}, Median: {stats['median']:.2f}, Q3: {stats['q3']:.2f}, Max: {stats['max']:.2f}"


class TemporalModels:
    def __init__(self, predictions, start_date, end_date):
        self.predictions = predictions.reset_index()
        self.start_date = start_date
        self.end_date = end_date

    def predict(self, X):
        X_tmp = X.reset_index()

        X_temporal = pd.DataFrame()
        X_temporal['year_month_day_hour'] = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq='h'
        )

        X_temporal = X_temporal.reset_index().merge(self.predictions,
                        how='left',
                        on='index')
        
        X_temporal = X_temporal.melt(id_vars=['year_month_day_hour'], 
                    value_vars=list(set(self.predictions.columns) - set(['index', 'year_month_day_hour'])), 
                    var_name='PULocationID', 
                    value_name='qty_travels')

        X_tmp = X_tmp.merge(X_temporal,
                            how='left',
                            on=['year_month_day_hour', 'PULocationID']).sort_values(by='index', ascending=True)

        return X_tmp['qty_travels']