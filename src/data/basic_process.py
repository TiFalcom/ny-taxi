import pandas as pd
import numpy as np
import logging
import click
import os
import yaml

from src.utils.transformers import FixFeaturesType, FixFeaturesMissing


@click.command()
@click.option('--config_file', default='features', type=str, help='Features configuration file on src/data/config.')
@click.option('--dataset_name', default=None, type=str, help='Data set name on data/raw.')
def main(config_file, dataset_name):

    logger = logging.getLogger('Basic-Process')
    
    logger.info(f'Loading {dataset_name} dataset.')
    df = pd.read_parquet(os.path.join('data', 'raw', f'{dataset_name}.parquet.gzip'))
    logger.info(f'Shape {dataset_name}: {df.shape}')

    logger.info(f'Loading weather dataset.')
    df_weather = pd.read_csv(os.path.join('data', 'external', 'weather_unify.csv'))
    logger.info(f'Shape weather: {df_weather.shape}')

    logger.info(f'Loading data-points dataset.')
    df_point = pd.read_parquet('data/external/Centerline_unify.parquet.gzip')
    logger.info(f'Shape point: {df_point.shape}')

    logger.info(f'Merging datasets.')

    # Weather
    df['year_month_day'] = df['pickup_datetime'].astype(str).str[0:10].str.replace('-', '')

    df_weather['year_month_day'] = df_weather['Year'].astype(str) + df_weather['Month'].astype(str).str.zfill(2) + df_weather['Day'].astype(str).str.zfill(2)

    df_interim = df.merge(df_weather[['year_month_day', 'daily_preciptation_normal_inches',
                                  'max_temperature_normal_f', 'min_temperature_normal_f',
                                  'avg_temperature_normal_f']], on='year_month_day', how='left')
    
    # Lat Long
    df_interim['lat_long'] = df_interim['pickup_longitude'].round(2).astype(str) + '_' + df_interim['pickup_latitude'].round(2).astype(str)

    df_point['lat_long'] = df_point['latitude'].round(2).astype(str) + '_' + df_point['longitude'].round(2).astype(str)

    df_point = df_point.drop_duplicates(subset=['lat_long'])

    df_interim = df_interim.merge(df_point, on='lat_long', how='left')

    logger.info(f'Datasets merged. Shape: {df_interim.shape}')

    logger.info(f'Removing travels from outside of NYC and non used columns.')
    west, south, east, north = -74.03, 40.63, -73.77, 40.85

    df_interim = df_interim[
        (df_interim['pickup_latitude'] >= south) &
        (df_interim['pickup_latitude'] <= north) &
        (df_interim['pickup_longitude'] >= west) &
        (df_interim['pickup_longitude'] <= east)
    ].reset_index(drop=True)

    if 'dropoff_datetime' in df_interim.columns:
        df_interim = df_interim.drop(columns=['dropoff_datetime', 'trip_duration'])

    # TODO: Add this features on src/data/config/features.yml
    df_interim = df_interim.drop(columns=['lat_long', 'latitude', 'CREATED', 'MODIFIED', 'FULL_STREE', 'ST_NAME',
                                          'longitude', 'PHYSICALID', 'L_LOW_HN', 'L_HIGH_HN', 'R_LOW_HN',
                                          'R_HIGH_HN', 'L_ZIP', 'R_ZIP', 'L_BLKFC_ID', 'R_BLKFC_ID', 'ST_LABEL',
                                          'STATUS', 'FRM_LVL_CO', 'TO_LVL_CO', 'SHAPE_Leng', 'PRE_MODIFI', 'POST_MODIF',
                                          'POST_DIREC'])
        
    logger.info(f'Cleaning complete. Shape: {df_interim.shape}.')

    logger.info(f'Fixing features with correct types.')
    cast_features = yaml.safe_load(open(os.path.join('src', 'data', 'config', f'{config_file}.yml'), 'r'))['cast_features']

    fix_features_type = FixFeaturesType(cast_features)

    df_interim = fix_features_type.transform(df_interim)
    logger.info(f'Fix completed. Shape: {df_interim.shape}')

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