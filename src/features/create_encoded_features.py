import pandas as pd
import logging
import click
import os
import yaml
import pickle
from src.utils.modeling import get_features_type, apply_encoders


@click.command()
@click.option('--config_file', default='features', type=str, help='Features configuration file on src/data/config.')
@click.option('--dataset_preffix', default=None, type=str, help='Data set name on data/raw.')
@click.option('--encoder_type', default=1, type=int, help='Type of encoder. 1-Ordinal Encoder 2-One Hot Encoder.')
def main(config_file, dataset_preffix, encoder_type):

    logger = logging.getLogger('Create-Encoded-Dataset')

    logger.info(f'Loading encoders.')

    selector = pickle.load(open(os.path.join(
            'models', 'encoders', 'selector.pkl'
        ), 'rb'))

    if encoder_type == 1:
        encoder = pickle.load(open(os.path.join(
                'models', 'encoders', 'ordinal_encoder.pkl'
            ), 'rb'))
    elif encoder_type == 2:
        encoder = pickle.load(open(os.path.join(
            'models', 'encoders', 'onehot_encoder.pkl'
        ), 'rb'))

    for table in ['train', 'test', 'valid']:
        logger.info(f'Loading {dataset_preffix} {table} dataset.')
        df = pd.read_parquet(os.path.join('data', 'processed', f'{dataset_preffix}_{table}.parquet.gzip'))
        logger.info(f'Shape {dataset_preffix} {table}: {df.shape}')

        # TODO: make a step to keep features configured on config_file
        logger.info(f'Starting Encoding')
        df = apply_encoders(df, [selector, encoder])
        logger.info(f'Encoding Completed! Shape: {df.shape}')

        logger.info('Saving dataset on data/processed_encoded')
        df.to_parquet(
            os.path.join('data', 'processed_encoded', f'{dataset_preffix}_{table}.parquet.gzip'),
            compression='gzip',
            index=False
        )
        logger.info('Saved!')

    logger.info('Success, all datasets encoded!')

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
        #filename='data/logs/basic_process.log',
        #filemode='w'
    )
    main()