import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from preprocessing import preprocess
from utils import read_male, read_female, read_train, read_test, build_feature_path

if __name__ == '__main__':
    male_lexicon = read_male()
    female_lexicon = read_female()
    male_cv = CountVectorizer(vocabulary=list(set(male_lexicon)))
    female_cv = CountVectorizer(vocabulary=list(set(female_lexicon)))

    train = read_train()
    test = read_test()

    for dataset, dataset_key in [(train, 'TRAINING_REL'),
                                 (test, 'TEST_REL')]:
        dataset['preprocessed_text'] = dataset.text.apply(preprocess)
        print('processing', dataset_key)
        feature_store_path = build_feature_path(dataset_key, 'male_words')
        male_matrix = male_cv.fit_transform(dataset.preprocessed_text).todense()
        df = pd.DataFrame(male_matrix, index=dataset.index, columns=male_cv.get_feature_names())
        df.to_csv(feature_store_path)
        feature_store_path = build_feature_path(dataset_key, 'male_words_total')
        df = df.sum(axis=1)
        df.to_csv(feature_store_path)

        feature_store_path = build_feature_path(dataset_key, 'female_words')
        female_matrix = female_cv.fit_transform(dataset.preprocessed_text).todense()
        df = pd.DataFrame(female_matrix, index=dataset.index, columns=female_cv.get_feature_names())
        df.to_csv(feature_store_path)
        feature_store_path = build_feature_path(dataset_key, 'female_words_total')
        df = df.sum(axis=1)
        df.to_csv(feature_store_path)
