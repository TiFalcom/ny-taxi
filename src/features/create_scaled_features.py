import pandas as pd
import logging
import click
import os
import yaml
import pickle
from src.utils.modeling import apply_encoders


@click.command()
@click.option('--config_file', default='features', type=str, help='Features configuration file on src/data/config.')
@click.option('--dataset_prefix', default=None, type=str, help='Data set name on data/raw.')
def main(config_file, dataset_prefix):

    logger = logging.getLogger('Create-Scaled-Dataset')

    logger.info(f'Loading Scalers.')

    scaler = pickle.load(open(os.path.join(
            'models', 'encoders', 'scaler.pkl'
        ), 'rb'))

    for table in ['valid', 'test', 'train']:
        logger.info(f'Loading {dataset_prefix} {table} dataset.')
        df = pd.read_parquet(os.path.join('data', 'aggregated', f'{dataset_prefix}_{table}.parquet.gzip'))
        logger.info(f'Shape {dataset_prefix} {table}: {df.shape}')

        logger.info(f'Starting Encoding')
        df = apply_encoders(df, [scaler])
        logger.info(f'Encoding Completed! Shape: {df.shape}')

        logger.info('Saving dataset on data/scaled')
        df.to_parquet(
            os.path.join('data', 'scaled', f'{dataset_prefix}_{table}.parquet.gzip'),
            compression='gzip',
            index=False
        )
        logger.info('Saved!')

    logger.info('Success, all datasets scaled!')

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
        #filename='data/logs/basic_process.log',
        #filemode='w'
    )
    main()