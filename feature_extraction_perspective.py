import re
from collections import defaultdict

import pandas as pd

from perspective import parse_summary_scores, get_toxicity_scores, REQUESTED_ATTRIBUTES_ALL
from utils import read_config, read_test, read_train, read_perspective_key, build_feature_path, read_gendered

gender_words = read_gendered()
gender_word_re = re.compile(r'(\b|^)(?P<gender>' + '|'.join(i.strip() for i in gender_words) + r')(\b|$)')


def perspective_df(input_df, key, languages):
    df = pd.DataFrame(map(parse_summary_scores,
                          get_toxicity_scores(input_df.text, key, languages=languages,
                                              requested_attributes=REQUESTED_ATTRIBUTES_ALL)), index=input_df.index)
    return df


def mask_gender_words(text, pattern=gender_word_re, replace_dict=defaultdict(lambda: "<MASK>")):
    spans = list()
    for match in re.finditer(pattern, text):
        spans.append(match.span('gender'))

    new_text = ""
    old_end = 0
    for start, end in spans:
        new_text += text[old_end:start]
        new_text += replace_dict[text[start:end]]
        old_end = end
    new_text += text[old_end:]
    return new_text


def compute_difference():
    for dataset_key in ['TRAINING_REL', 'TEST_REL']:
        original_path = build_feature_path(dataset_key, 'perspective')
        masked_path = build_feature_path(dataset_key, 'perspective_masked')
        feature_store_path = build_feature_path(dataset_key, 'perspective_difference')
        original_df = pd.read_csv(original_path, index_col='id')
        masked_df = pd.read_csv(masked_path, index_col='id')
        for col in original_df.columns:
            original_df[col] -= masked_df[col]
        original_df.to_csv(feature_store_path)


if __name__ == '__main__':

    config = read_config()
    train = read_train()
    test = read_test()
    print(train[['text', 'language']].sample(10))
    key = read_perspective_key()
    mask_gendered = True
    if mask_gendered:
        train['text'] = train.text.apply(mask_gender_words)
        test['text'] = test.text.apply(mask_gender_words)
        print(train[['text', 'language']].sample(10))
    for dataset, dataset_key in [(train, 'TRAINING_REL'),
                                 (test, 'TEST_REL')]:
        print('processing', dataset_key)
        feature_store_path = build_feature_path(dataset_key, mask_gendered and 'perspective_masked' or 'perspective')
        df = perspective_df(dataset, key, ['en'])
        df.to_csv(feature_store_path)
    if mask_gendered:
        compute_difference()
