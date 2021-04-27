import json
import pandas as pd
import os


def read_config():
    with open('config.json') as f:
        config = json.load(f)
    return config


def read_train():
    df = __read_data('TRAINING_REL', sep='\t',
                     usecols=['id', 'source', 'language', 'text', 'task1', 'task2'])
    return df.set_index('id')


def read_test():
    df = __read_data('TEST_REL', sep='\t',
                     usecols=['id', 'source', 'language', 'text'])
    return df.set_index('id')


def read_male():
    df = __read_data('MALE_VOCAB_REL')
    return df.values.ravel().tolist()


def read_female():
    df = __read_data('FEMALE_VOCAB_REL')
    return df.values.ravel().tolist()


def read_gendered():
    df = __read_data('GENDERED_VOCAB_REL')
    return df.values.ravel().tolist()


def __read_data(rel_path_key, usecols=None, sep='\t'):
    config = read_config()
    df = pd.read_csv(os.path.join(config['DATA_ROOT'], config[rel_path_key]), sep=sep,
                     usecols=usecols)
    return df
