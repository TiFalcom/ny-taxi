import pandas as pd
import logging
import click
import os


@click.command()
@click.option('--datasets_list', default=None, multiple=True, help='List of datasets with full path to be concat.')
@click.option('--dataset_name_output', default=None, type=str, help='Name of the output dataset.')
def main(datasets_list, dataset_name_output):

    logger = logging.getLogger('Data-Concat')
    
    lst_df = []
    for dataset in datasets_list:

        logger.info(f'Loading {dataset}.')
        lst_df.append(pd.read_parquet(os.path.join(f'{dataset}.parquet.gzip')))
        logger.info(f'Shape {dataset}: {lst_df[-1].shape}')

    logger.info(f'Concating datasets.')
    df = pd.concat(lst_df)
    logger.info(f'Shape {dataset_name_output}: {df.shape}')

    logger.info('Saving dataset.')
    df.to_parquet(
        os.path.join(f'{dataset_name_output}.parquet.gzip'),
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