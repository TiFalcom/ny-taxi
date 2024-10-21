import pandas as pd
import logging
import click
import os
import yaml
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.preprocessing import StandardScaler
import pickle


@click.command()
@click.option('--config_file', default='features', type=str, help='Features configuration file on src/data/config.')
@click.option('--dataset_prefix', default=None, type=str, help='Data set name on data/raw.')
def main(config_file, dataset_prefix):

    logger = logging.getLogger('Normalize-Features')

    # TODO: fix hard_remove to use here
    config_features = yaml.safe_load(open(os.path.join('src', 'data', 'config', f'{config_file}.yml'), 'r'))['aggregate']
    
    features_to_normalize = ['period_of_day_dawn_sum','Maximum_max','period_of_day_morning_sum',
                             'Minimum_max','day_of_week_max','period_of_day_evening_sum',
                             'period_of_day_afternoon_sum','passenger_count_sum']

    logger.info(f'Loading {dataset_prefix} train dataset.')
    df = pd.read_parquet(os.path.join('data', 'aggregated', f'{dataset_prefix}_train.parquet.gzip'))
    logger.info(f'Shape {dataset_prefix} train: {df.shape}')

    scaler = SklearnTransformerWrapper(StandardScaler(), variables=features_to_normalize)

    scaler = scaler.fit(df)

    logger.info(f'Saving scaler binary.')

    pickle.dump(scaler, open(os.path.join(
        'models', 'encoders', 'scaler.pkl'
    ), 'wb'))

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