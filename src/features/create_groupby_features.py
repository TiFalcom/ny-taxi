import pandas as pd
import logging
import click
import os
import yaml
from src.utils.modeling import p75, p99


@click.command()
@click.option('--config_file', default='features', type=str, help='Features configuration file on src/data/config.')
@click.option('--dataset_prefix', default=None, type=str, help='Data set name on data/raw.')
@click.option('--dataset_output_prefix', default=None, type=str, help='Data set output name data/raw.')
def main(config_file, dataset_prefix, dataset_output_prefix):

    logger = logging.getLogger('Aggregate-Features')

    if not dataset_output_prefix:
        dataset_output_prefix = dataset_prefix

    config_features = yaml.safe_load(open(os.path.join('src', 'data', 'config', f'{config_file}.yml'), 'r'))['aggregate']
    aggregate_col = config_features['temporal_feature']
    aggregate_features = config_features['aggregate_features']

    for table in ['valid', 'test', 'train']:
        logger.info(f'Loading {dataset_prefix} {table} dataset.')
        df = pd.read_parquet(os.path.join('data', 'processed_encoded', f'{dataset_prefix}_{table}.parquet.gzip'))
        logger.info(f'Shape {dataset_prefix} {table}: {df.shape}')

        logger.info('Aggregating features.')

        df_agg = df.groupby(aggregate_col).agg(aggregate_features).reset_index()

        df_agg.columns = [f'{col[0]}_{col[1]}' if col[1] != '' else f'{col[0]}' 
                          for col in df_agg.columns]

        # TODO: Move to another script and add to config_file
        df_agg['qty_travels'] = df_agg['VendorID_count'].shift(-1)

        df_agg = df_agg.iloc[:-1].reset_index(drop=True).drop(columns='VendorID_count')

        logger.info(f'Aggregated: {df_agg.shape}')

        logger.info('Saving dataset on data/aggregated')
        df_agg.to_parquet(
            os.path.join('data', 'aggregated', f'{dataset_output_prefix}_{table}.parquet.gzip'),
            compression='gzip',
            index=False
        )
        logger.info('Saved!')

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