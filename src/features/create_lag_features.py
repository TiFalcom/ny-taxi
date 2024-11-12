import pandas as pd
import logging
import click
import os
from sklearn.cluster import KMeans
from src.utils.transformers import FeatureEngineeringLag


@click.command()
@click.option('--dataset_prefix', default=None, type=str, help='Data set name on data/raw.')
def main(dataset_prefix):

    logger = logging.getLogger('Feature-Engineering-Lag')

    create_features = FeatureEngineeringLag()

    lst_df = []

    for dataset in ['train', 'test', 'valid']:
        logger.info(f'Loading {dataset_prefix} dataset.')
        lst_df.append(pd.read_parquet(os.path.join('data', 'aggregated', f'{dataset_prefix}_{dataset}.parquet.gzip')).reset_index(drop=True))
        logger.info(f'Shape {dataset_prefix} {dataset}: {lst_df[-1].shape}')

    df = pd.concat(lst_df)

    logger.info(f'Starting Feature Engineering')
    df = create_features.transform(df)
    logger.info(f'Feature Enginerring Lag Completed! Shape: {df.shape}')

    logger.info('Saving datasets on data/lag')

    for index, dataset in enumerate(['train', 'test', 'valid']):
        lst_df[index] = lst_df[index][['year_month_day_hour', 'PULocationID']].merge(df, how='left', on=['year_month_day_hour', 'PULocationID']).reset_index(drop=True)
        lst_df[index].to_parquet(
            os.path.join('data', 'lag', f'{dataset_prefix}_{dataset}.parquet.gzip'),
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