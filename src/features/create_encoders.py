import pandas as pd
import logging
import click
import os
import yaml
import pickle
from src.utils.transformers import Selector
from feature_engine.encoding import OrdinalEncoder, OneHotEncoder
from src.utils.modeling import get_features_type


@click.command()
@click.option('--config_file', default='features', type=str, help='Features configuration file on src/data/config.')
@click.option('--dataset_preffix', default=None, type=str, help='Data set name on data/raw.')
def main(config_file, dataset_preffix):

    logger = logging.getLogger('Create-Encoders')

    logger.info(f'Loading {dataset_preffix} train dataset.')
    df = pd.read_parquet(os.path.join('data', 'processed', f'{dataset_preffix}_train.parquet.gzip'))
    logger.info(f'Shape {dataset_preffix} train: {df.shape}')

    logger.info(f'Loading config file {config_file}')

    config_features = yaml.safe_load(open(os.path.join('src', 'data', 'config', f'{config_file}.yml'), 'r'))

    features_selected = list(set(df.columns) - set(config_features['hard_remove']))

    categorical_features, _ = get_features_type(df, 15, 
                                                config_features['hard_remove'] + [config_features['temporal_feature']])

    logger.info(f'Features selected: {features_selected}')
    logger.info(f'Categorical features: {categorical_features}')

    logger.info(f'Starting encoders creation.')
    # Feature Selection
    selector = Selector(features_selected)

    df = selector.transform(df)

    # Ordinal Encoder
    ord_enc = OrdinalEncoder(encoding_method='arbitrary',
                             variables=categorical_features,
                             missing_values='ignore',
                             ignore_format=True).fit(df)

    # One Hot Encoder
    one_encoder = OneHotEncoder(#drop_last=True, 
                            #drop_last_binary=True,
                            variables=categorical_features,
                            ignore_format=True).fit(df)
    
    logger.info(f'Saving encoders binary.')

    pickle.dump(selector, open(os.path.join(
        'models', 'encoders', 'selector.pkl'
    ), 'wb'))

    pickle.dump(ord_enc, open(os.path.join(
        'models', 'encoders', 'ordinal_encoder.pkl'
    ), 'wb'))

    pickle.dump(one_encoder, open(os.path.join(
        'models', 'encoders', 'onehot_encoder.pkl'
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