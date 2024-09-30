import pandas as pd
import numpy as np
import logging
import click
import os
import yaml

from src.utils.transformers import FixFeatures


@click.command()
@click.option('--dataset_name_train', default=None, type=str, help='Data set train name on data/interim.')
@click.option('--dataset_name_test', default=None, type=str, help='Data set test name on data/interim.')
def main(dataset_name_train, dataset_name_test):

    logger = logging.getLogger('Data-Concat')
    
    logger.info(f'Loading {dataset_name_train} dataset.')
    df = pd.read_parquet(os.path.join('data', 'interim', f'{dataset_name_train}.parquet.gzip'))
    logger.info(f'Shape {dataset_name_train}: {df.shape}')

    logger.info(f'Loading {dataset_name_test} dataset.')
    df_test = pd.read_parquet(os.path.join('data', 'interim', f'{dataset_name_test}.parquet.gzip'))
    logger.info(f'Shape {dataset_name_test}: {df_test.shape}')

    logger.info(f'Merging datasets.')

    df = pd.concat([df, df_test], ignore_index=True).reset_index(drop=True)

    logger.info('Saving dataset on data/interim')
    df.to_parquet(
        os.path.join('data', 'interim', f'full.parquet.gzip'),
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