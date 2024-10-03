import pandas as pd
import logging
import click
import os
import yaml
import zipfile
import shutil


@click.command()
@click.option('--source_file', default=None, type=str, help='Source Folder to retrieve data')
def main(source_file):

    logger = logging.getLogger('Retrieve-Data')

    logger.info(f'Retrieving data from {source_file}.')

    os.makedirs('temp', exist_ok=True)

    with zipfile.ZipFile(source_file, 'r') as zipf:
        zipf.extractall('temp')
        files = zipf.namelist()
        logger.info(f'Files to retrieve: {files}.')

    for file in files:
        logger.info(f'Retrieving {file}.')
        shutil.copy2(os.path.join('temp', file), file)

    shutil.rmtree('temp')

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