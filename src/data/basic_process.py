import pandas as pd
import numpy as np
import logging
import click
import os
import yaml
from datetime import datetime
import sys

from src.utils.transformers import FixFeaturesType, FixFeaturesMissing


@click.command()
@click.option('--config_file', default='features', type=str, help='Features configuration file on src/data/config.')
@click.option('--dataset_name', default=None, type=str, help='Data set name on data/raw.')
def main(config_file, dataset_name):

    logger = logging.getLogger('Basic-Process')
    
    logger.info(f'Loading {dataset_name} dataset.')
    df = pd.read_parquet(os.path.join('data', 'raw', f'{dataset_name}.parquet'))
    logger.info(f'Shape {dataset_name}: {df.shape}')

    logger.info(f'Loading weather dataset.')
    df_weather = pd.read_parquet('data/external/weather_unify.parquet.gzip')
    logger.info(f'Shape weather: {df_weather.shape}')

    logger.info(f'Loading data-points dataset.')
    df_point = pd.read_csv('data/external/taxi_zone_lookup.csv', delimiter=',')
    logger.info(f'Shape point: {df_point.shape}')

    logger.info(f'Merging datasets.')

    # Weather
    df['year_month_day'] = df['tpep_pickup_datetime'].astype(str).str[0:10].str.replace('-', '')
    df_weather['year_month_day'] = df_weather['Date'].apply(lambda x: datetime.strftime(datetime.strptime(x, '%d/%m/%Y'), '%Y%m%d'))

    df_interim = df.merge(df_weather[['year_month_day', 'Maximum', 'Minimum', 
                                  'Average', 'Precipitation', 'new_snow', 
                                  'snow_depth']], on='year_month_day', how='left')
    
    # Lat Long
    df_interim = df_interim.merge(df_point, right_on='LocationID', left_on='PULocationID', how='left')

    df_interim = df_interim.merge(df_point, right_on='LocationID', 
                              left_on='DOLocationID', how='left', suffixes=('_PU', '_DO'))

    logger.info(f'Datasets merged. Shape: {df_interim.shape}')
        
    logger.info(f'Cleaning complete. Shape: {df_interim.shape}.')

    #logger.info(f'Fixing features with correct types.')
    #cast_features = yaml.safe_load(open(os.path.join('src', 'data', 'config', f'{config_file}.yml'), 'r'))['cast_features']

    #fix_features_type = FixFeaturesType(cast_features)

    #df_interim = fix_features_type.transform(df_interim)
    #logger.info(f'Fix completed. Shape: {df_interim.shape}')

    logger.info(f'Fixing features with missing values.')
    missing_features = yaml.safe_load(open(os.path.join('src', 'data', 'config', f'{config_file}.yml'), 'r'))['missing_features']

    fix_features_missing = FixFeaturesMissing(missing_features)

    df_interim = fix_features_missing.transform(df_interim)
    logger.info(f'Fix completed. Shape: {df_interim.shape}')

    logger.info('Saving dataset on data/interim')
    df_interim.to_parquet(
        os.path.join('data', 'interim', f'{dataset_name}.parquet.gzip'),
        compression='gzip',
        index=False
    )
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