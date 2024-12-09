import pandas as pd
import logging
import click
import os
import yaml
from sklearn.tree import DecisionTreeRegressor
import pickle
from src.utils.modeling import get_metrics, get_boxplot_stats

pd.set_option('display.max_columns', None)

@click.command()
@click.option('--config_file', default='features', type=str, help='Features configuration file on src/data/config.')
@click.option('--folder_dataset_prefix', default=None, type=str, help='Folder with data set name on data/raw.')
@click.option('--hyperparams_file', default='hyperparams', type=str, help='Hyperparams config file.')
@click.option('--model_suffix', default='default', type=str, help='Hyperparams config file.')
def main(config_file, folder_dataset_prefix, hyperparams_file, model_suffix):

    logger = logging.getLogger('Train-Boosting')

    logger.info('Loading Training Dataset')

    logger.info(f'Loading {folder_dataset_prefix} train dataset.')
    df_train = pd.read_parquet(os.path.join('data', f'{folder_dataset_prefix}_train.parquet.gzip'))
    logger.info(f'Shape {folder_dataset_prefix} train: {df_train.shape}')

    logger.info(f'Loading {folder_dataset_prefix} valid dataset.')
    df_valid = pd.read_parquet(os.path.join('data', f'{folder_dataset_prefix}_valid.parquet.gzip'))
    logger.info(f'Shape {folder_dataset_prefix} valid: {df_valid.shape}')

    df_train = df_train[df_train['PULocationID'].isin([163, 79])].reset_index(drop=True)
    df_valid = df_valid[df_valid['PULocationID'].isin([163, 79])].reset_index(drop=True)

    params = yaml.safe_load(open(os.path.join('src', 'model', 'config', f'{hyperparams_file}.yml'), 'r'))['decision_tree']

    logger.info(f'Decision Tree will be fitted with the following hyperparams: {params}')

    config_features = yaml.safe_load(open(os.path.join('src', 'data', 'config', f'{config_file}.yml'), 'r'))

    target = config_features['target'][0]
    features = list(set(df_train.columns) - set(config_features['hard_remove'] + 
                                          params['features_hard_remove'] + [target]))

    logger.info(f'Features: {features}')
    logger.info(f'Target: {target}')

    model = DecisionTreeRegressor(
        **params['params']
    )

    logger.info(f'Training model')

    model.fit(df_train[features], df_train[target])

    logger.info(f'Model trained! Evaluating...')

    print(
        get_metrics(
            [
                (model, model.feature_names_in_, 'with lag features', df_train, df_valid)
            ],
            df_train,
            df_valid,
            target
        )
    )

    df_train['pred'] = model.predict(df_train[features])

    df_valid['pred'] = model.predict(df_valid[features])

    print(get_boxplot_stats(
        df_valid[target],
        'True   '
    ))

    print(get_boxplot_stats(
        df_valid['pred'],
        'Predict'
    ))

    logger.info('Saving binary model.')

    pickle.dump(model, open(os.path.join('models', 'models', f'dt_{model_suffix}.pkl'), 'wb'))

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