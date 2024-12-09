import pandas as pd
import logging
import click
import os
import yaml
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
from src.utils.modeling import get_metrics, get_boxplot_stats, TemporalModels
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)


@click.command()
@click.option('--config_file', default='features', type=str, help='Features configuration file on src/data/config.')
@click.option('--folder_dataset_prefix', default=None, type=str, help='Folder with data set name on data/raw.')
@click.option('--hyperparams_file', default='hyperparams', type=str, help='Hyperparams config file.')
@click.option('--model_suffix', default='default', type=str, help='Hyperparams config file.')
def main(config_file, folder_dataset_prefix, hyperparams_file, model_suffix):

    logger = logging.getLogger('Train-Sarimax')

    logger.info('Loading Training Dataset')

    logger.info(f'Loading {folder_dataset_prefix} train dataset.')
    df_train = pd.read_parquet(os.path.join('data', f'{folder_dataset_prefix}_train.parquet.gzip'))
    logger.info(f'Shape {folder_dataset_prefix} train: {df_train.shape}')

    logger.info(f'Loading {folder_dataset_prefix} valid dataset.')
    df_valid = pd.read_parquet(os.path.join('data', f'{folder_dataset_prefix}_valid.parquet.gzip'))
    logger.info(f'Shape {folder_dataset_prefix} valid: {df_valid.shape}')

    # Dataset fix
    df_temporal = pd.DataFrame()
    df_temporal['year_month_day_hour'] = pd.date_range(
        start=df_train['year_month_day_hour'].min(),
        end=df_valid['year_month_day_hour'].max(),
        freq='h'
    )

    df_train['year_month_day_hour'] = pd.to_datetime(df_train['year_month_day_hour'], format='%Y-%m-%d %H')
    df_valid['year_month_day_hour'] = pd.to_datetime(df_valid['year_month_day_hour'], format='%Y-%m-%d %H')

    df_train = df_train[df_train['PULocationID'].isin([163, 79])].reset_index(drop=True)
    df_valid = df_valid[df_valid['PULocationID'].isin([163, 79])].reset_index(drop=True)

    params = yaml.safe_load(open(os.path.join('src', 'model', 'config', f'{hyperparams_file}.yml'), 'r'))['sarimax']

    logger.info(f'SARIMAX will be fitted with the following hyperparams: {params}')

    config_features = yaml.safe_load(open(os.path.join('src', 'data', 'config', f'{config_file}.yml'), 'r'))

    target = config_features['target'][0]

    # need to implement arima recursive fit 
    # https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_forecasting.html
    
    logger.info(f'Training model')

    locations_forecasts = {}
    for location in tqdm([163, 79]):#, 100, 140, 107, 186, 48, 132]):

        df = pd.concat([df_train, df_valid]).sort_values(by='year_month_day_hour', ascending=True).reset_index(drop=True)

        df = (
            df_temporal
            .merge(df[['PULocationID', 
                             'qty_travels', 
                             'year_month_day_hour']
                           ][df['PULocationID'] == location],
                             how='left', 
                             on=['year_month_day_hour'],
                             suffixes=('_', '')
            )[['year_month_day_hour','qty_travels']]
            .fillna(0)
            .sort_values(by='year_month_day_hour', ascending=True)
            #.set_index('year_month_day_hour')
        )['qty_travels']

        #df_train_location.index.freq = 'h'

        #df_valid_location = (
        #    df_temporal
        #    .merge(df_valid[['PULocationID', 
        #                     'qty_travels', 
        #                     'year_month_day_hour']
        #                   ][df_valid['PULocationID'] == location],
        #                     how='left', 
        #                     on=['year_month_day_hour'],
        #                     suffixes=('_', '')
        #    )[['year_month_day_hour','qty_travels']]
        #    .fillna(0)
        #    .sort_values(by='year_month_day_hour', ascending=True)
        #    #.set_index('year_month_day_hour')
        #)['qty_travels']
        #df_valid_location.index = [df_train_location.shape[0] + i for i in range(df_valid_location.shape[0])]

        #df_valid_location.index.freq = 'h'
    
        forecasts = []
        for i in range(744, df.shape[0]+1):
            sarimax = SARIMAX(df.iloc[i-24:i], order=tuple(params['params']['order']), seasonal_order=tuple(params['params']['seasonal_order']))#, freq='h')
            sarimax = sarimax.fit()
            forecasts += list(sarimax.forecast(steps=1).values)

        locations_forecasts[location] = forecasts

    df_forecast = pd.DataFrame(locations_forecasts)

    df_forecast.to_parquet('data/forecast/top8_sarimax.parquet.gzip', compression='gzip', index=False)

    pickle.dump(locations_forecasts, open('models/models/sarimax.pkl', 'wb'))

    logger.info(f'Model trained! Evaluating...')

    model = TemporalModels(df_forecast, start_date='2024-02-01 00', end_date='2024-02-07 23')

    print(
        get_metrics(
            [
                (model, ['year_month_day_hour', 'PULocationID'], 'sarimax top8', df_valid, df_valid)
            ],
            df_valid,
            df_valid,
            target
        )
    )

    #df_train['pred'] = model.predict(df_train[['year_month_day_hour', 'PULocationID']])

    df_valid['pred'] = model.predict(df_valid[['year_month_day_hour', 'PULocationID']])

    print(get_boxplot_stats(
        df_valid[target],
        'True   '
    ))

    print(get_boxplot_stats(
        df_valid['pred'],
        'Predict'
    ))

    logger.info('Saving binary model.')

    pickle.dump(model, open(os.path.join('models', 'models', f'sarimax_{model_suffix}.pkl'), 'wb'))

    logger.info('Success!')


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
        #filename='data/logs/basic_process.log',
        #filemode='w'
    )
    main()