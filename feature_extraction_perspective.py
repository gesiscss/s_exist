import pandas as pd

from perspective import parse_summary_scores, get_toxicity_scores, REQUESTED_ATTRIBUTES_ALL
from utils import read_config, read_test, read_train, read_perspective_key, build_feature_path


def perspective_df(input_df, key, languages):
    df = pd.DataFrame(map(parse_summary_scores,
                          get_toxicity_scores(input_df.text, key, languages=languages,
                                              requested_attributes=REQUESTED_ATTRIBUTES_ALL)), index=input_df.index)
    return df


if __name__ == '__main__':

    config = read_config()
    train = read_train()
    test = read_test()
    print(train[['text', 'language']].sample(10))
    key = read_perspective_key()

    for dataset, dataset_key in [(train, 'TRAINING_REL'),
                                 (test, 'TEST_REL')]:
        print('processing', dataset_key)
        feature_store_path = build_feature_path(dataset_key, 'perspective')
        df = perspective_df(dataset, key, ['en'])
        df.to_csv(feature_store_path)
