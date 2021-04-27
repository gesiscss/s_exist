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


def read_sexism(with_modifications=False):
    df = __read_data('SEXISM_REL', sep=',',
                     usecols=['id', 'dataset', 'text', 'toxicity', 'sexist', 'sexist_content', 'sexist_phrasing',
                              'scale', 'of_id'])
    df = df.set_index('id')
    if not with_modifications:
        df = df[df.of_id == -1]
    return df


if __name__ == '__main__':
    print(read_sexism(with_modifications=False).head())


def read_perspective_key():
    config = read_config()
    PERSPECTIVE_API_PATH = os.path.join(config['DATA_ROOT'], config['PERSPECTIVE_API_REL'])
    with open(PERSPECTIVE_API_PATH) as f:
        key = f.read().strip()
    return key


def build_feature_path(dataset_key, feature_name):
    config = read_config()
    return os.path.join(config['DATA_ROOT'], config[dataset_key], feature_name)
