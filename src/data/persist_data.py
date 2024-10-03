import pandas as pd
import logging
import click
import os
import yaml
import zipfile


@click.command()
@click.option('--config_file', default='data', type=str, help='Features configuration file on src/data/config.')
@click.option('--target_file', default=None, type=str, help='Target Folder to replicate data')
def main(config_file, target_file):

    logger = logging.getLogger('Persist-Data')

    list_datasets = yaml.safe_load(open(os.path.join('src', 'data', 'config', f'{config_file}.yml'), 'r'))['files_to_persist']

    logger.info(f'Datasets that will be persisted on {target_file}: {list_datasets}')

    logger.info(f'Zipping files')

    with zipfile.ZipFile(target_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for arquivo in list_datasets:
            logger.info(f'Zipping {arquivo}')
            zipf.write(arquivo, arquivo)

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