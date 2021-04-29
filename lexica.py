import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from preprocessing import preprocess
from utils import read_hedges, read_boosters, read_train, read_test, build_feature_path

if __name__ == '__main__':
    hedges_lexicon = read_hedges()
    boosters_lexicon = read_boosters()
    hedges_cv = CountVectorizer(vocabulary=list(set(hedges_lexicon)))
    boosters_cv = CountVectorizer(vocabulary=list(set(boosters_lexicon)))

    train = read_train()
    test = read_test()

    for dataset, dataset_key in [(train, 'TRAINING_REL'),
                                 (test, 'TEST_REL')]:
        dataset['preprocessed_text'] = dataset.text.apply(preprocess)
        print('processing', dataset_key)
        feature_store_path = build_feature_path(dataset_key, 'hedges')
        hedges_matrix = hedges_cv.fit_transform(dataset.preprocessed_text).todense()
        df = pd.DataFrame(hedges_matrix, index=dataset.index, columns=hedges_cv.get_feature_names())
        df.to_csv(feature_store_path)
        feature_store_path = build_feature_path(dataset_key, 'hedges_total')
        df = df.sum(axis=1)
        df.to_csv(feature_store_path)

        feature_store_path = build_feature_path(dataset_key, 'boosters')
        boosters_matrix = boosters_cv.fit_transform(dataset.preprocessed_text).todense()
        df = pd.DataFrame(boosters_matrix, index=dataset.index, columns=boosters_cv.get_feature_names())
        df.to_csv(feature_store_path)
        feature_store_path = build_feature_path(dataset_key, 'boosters_total')
        df = df.sum(axis=1)
        df.to_csv(feature_store_path)
