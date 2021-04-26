import json
import pandas as pd
import os


def read_config():
    with open('config.json') as f:
        config = json.load(f)
    return config


def read_train():
    config = read_config()
    df = pd.read_csv(os.path.join(config['DATA_ROOT'], config['TRAINING_REL']), sep='\t',
                     usecols=['id', 'source', 'language', 'text', 'task1', 'task2']).set_index('id')
    return df


def read_test():
    config = read_config()
    df = pd.read_csv(os.path.join(config['DATA_ROOT'], config['TEST_REL']), sep='\t',
                     usecols=['id', 'source', 'language', 'text']).set_index('id')
    return df
