import pandas as pd

import re
import string

from sklearn.feature_extraction.text import CountVectorizer

from preprocessing import preprocess, replace_pattern
from utils import read_train, read_test, build_feature_path

hashtag_pattern = re.compile(r'(^|\W)(?P<hashtag>#\w+)(\b|[' + string.punctuation + '])', flags=re.I | re.M | re.DOTALL)
mention_pattern = re.compile(r'(^|\W)(?P<mention>(rt|ht|cc|[.] ?)?(@\w+|MENTION\d+))(\b|[' + string.punctuation + '])',
                             flags=re.I | re.M | re.DOTALL)

if __name__ == '__main__':

    train = read_train()
    test = read_test()

    all_hashtags = train['text'].str.extractall(r'(\#\w+)')[0].value_counts().keys()
    all_mentions = train['text'].str.extractall(r'(\@\w+)')[0].value_counts().keys()

    hashtags_cv = CountVectorizer(vocabulary=list(set(all_hashtags)), token_pattern = r'\b\w\w+\b|(?<!\w)@\w+|(?<!\w)#\w+')
    mentions_cv = CountVectorizer(vocabulary=list(set(all_mentions)), token_pattern = r'\b\w\w+\b|(?<!\w)@\w+|(?<!\w)#\w+')

    for dataset, dataset_key in [(train, 'TRAINING_REL'),
                                 (test, 'TEST_REL')]:
        
    
        mention_list = list()
        hashtag_list = list()

        for text in dataset['text'].values:
            mention_list.append(len(re.findall(mention_pattern, text)))
            hashtag_list.append(len(re.findall(hashtag_pattern, text)))

        dataset['hashtag_count'] = hashtag_list
        dataset['mention_count'] = mention_list

        feature_store_path = build_feature_path(dataset_key, 'hashtag_count')
        df = dataset[['hashtag_count']]
        df.to_csv(feature_store_path)

        feature_store_path = build_feature_path(dataset_key, 'mention_count')
        df = dataset[['mention_count']]
        df.to_csv(feature_store_path)

        feature_store_path = build_feature_path(dataset_key, 'hashtags')
        hashtags_matrix = hashtags_cv.fit_transform(dataset.text).todense()
        df = pd.DataFrame(hashtags_matrix, index=dataset.index, columns=hashtags_cv.get_feature_names())
        df.to_csv(feature_store_path)
        
        feature_store_path = build_feature_path(dataset_key, 'hashtags_total')
        df = df.sum(axis=1)
        df.to_csv(feature_store_path)


        feature_store_path = build_feature_path(dataset_key, 'mentions')
        mentions_matrix = mentions_cv.fit_transform(dataset.text).todense()
        df = pd.DataFrame(mentions_matrix, index=dataset.index, columns=mentions_cv.get_feature_names())
        df.to_csv(feature_store_path)
        
        feature_store_path = build_feature_path(dataset_key, 'mentions_total')
        df = df.sum(axis=1)
        df.to_csv(feature_store_path)