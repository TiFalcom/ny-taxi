import pandas as pd
import numpy as np
import logging
import click
import os
import yaml
import pickle
from sklearn.cluster import KMeans
from src.utils.transformers import FeatureEngineering


@click.command()
@click.option('--dataset_preffix', default=None, type=str, help='Data set name on data/raw.')
def main(dataset_preffix):

    logger = logging.getLogger('Feature-Engineering')
    
    logger.info(f'Training K-means with valid dataset.')

    logger.info(f'Loading {dataset_preffix} valid dataset.')
    df = pd.read_parquet(os.path.join('data', 'train_test', f'{dataset_preffix}_valid.parquet.gzip'))
    logger.info(f'Shape {dataset_preffix} valid: {df.shape}')

    longitude = list(df['pickup_longitude']) + list(df['dropoff_longitude'])
    latitude = list(df['pickup_latitude']) + list(df['dropoff_latitude'])

    loc_df = pd.DataFrame()
    loc_df['longitude'] = longitude
    loc_df['latitude'] = latitude

    loc_df = loc_df.astype(np.float32).values

    # Defined with experiments
    # TODO: Remove from this file, need to transfer to a specific script
    kmeans = KMeans(n_clusters=6, random_state=777, algorithm='lloyd').fit(loc_df)

    create_features = FeatureEngineering(kmeans)

    for table in ['train', 'test', 'valid']:
        logger.info(f'Loading {dataset_preffix} {table} dataset.')
        df = pd.read_parquet(os.path.join('data', 'train_test', f'{dataset_preffix}_{table}.parquet.gzip'))
        logger.info(f'Shape {dataset_preffix} {table}: {df.shape}')

        logger.info(f'Starting Feature Engineering')
        df = create_features.transform(df)
        logger.info(f'Feature Enginerring Completed! Shape: {df.shape}')

        logger.info('Saving dataset on data/processed')
        df.to_parquet(
            os.path.join('data', 'processed', f'{dataset_preffix}_{table}.parquet.gzip'),
            compression='gzip',
            index=False
        )
        logger.info('Saved!')

    logger.info('Saving K-means artifact!')

    pickle.dump(kmeans, open(os.path.join('models', 'encoders', 'kmeans.pkl'), 'wb'))

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