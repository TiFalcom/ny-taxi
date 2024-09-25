import pandas as pd
import numpy as np
import logging
import click
import os
import yaml
from datetime import datetime

@click.command()
@click.option('--config_file', default='features', type=str, help='Features configuration file on src/data/config.')
@click.option('--dataset_name', default=None, type=str, help='Data set name on data/interim.')
@click.option('--ymd_train', default=None, type=str, help='YYYY-MM-DD from beggining of the training dataset, included')
@click.option('--ymd_test', default=None, type=str, help='YYYY-MM-DD from beggining of the testing dataset, included')
def main(config_file, dataset_name, ymd_train, ymd_test):

    logger = logging.getLogger('Split-Data')
    
    ymd_train = datetime.strptime(ymd_train, '%Y%m%d')
    ymd_test = datetime.strptime(ymd_test, '%Y%m%d')

    temporal_feature = yaml.safe_load(open(os.path.join('src', 'data', 'config', f'{config_file}.yml'), 'r'))['temporal_feature']

    logger.info(f'Loading {dataset_name} dataset.')
    df = pd.read_parquet(os.path.join('data', 'interim', f'{dataset_name}.parquet.gzip'))
    logger.info(f'Shape {dataset_name}: {df.shape}')

    index_valid = df[(df[temporal_feature] >= ymd_train) & (df[temporal_feature] < ymd_test)].sample(frac=0.1, random_state=777).index
    index_train = df[(df[temporal_feature] >= ymd_train) & (df[temporal_feature] < ymd_test) & (df.index.isin(index_valid) == False)].index
    index_test = df[(df[temporal_feature] >= ymd_test)].index

    logger.info(f'Saving train dataset. Shape: {df.iloc[index_train].shape}')
    df.iloc[index_train].to_parquet(
        os.path.join('data', 'train_test', f'{dataset_name}_train.parquet.gzip'),
        compression='gzip',
        index=False
    )
    logger.info('Success!')

    logger.info(f'Saving test dataset. Shape: {df.iloc[index_test].shape}')
    df.iloc[index_test].to_parquet(
        os.path.join('data', 'train_test', f'{dataset_name}_test.parquet.gzip'),
        compression='gzip',
        index=False
    )
    logger.info('Success!')

    logger.info(f'Saving valid dataset. Shape: {df.iloc[index_valid].shape}')
    df.iloc[index_valid].to_parquet(
        os.path.join('data', 'train_test', f'{dataset_name}_valid.parquet.gzip'),
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