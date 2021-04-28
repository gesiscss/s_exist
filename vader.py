import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

from preprocessing import preprocess
from utils import read_train, read_test, build_feature_path

if __name__ == '__main__':
    train = read_train()
    test = read_test()

    for dataset, dataset_key in [(train, 'TRAINING_REL'),
                                 (test, 'TEST_REL')]:
        print('processing', dataset_key)
        dataset['preprocessed_text'] = dataset.text.apply(preprocess)
        feature_store_path = build_feature_path(dataset_key, 'vader')
        vader = SentimentIntensityAnalyzer()
        df = pd.DataFrame(dataset.preprocessed_text.apply(vader.polarity_scores).values.tolist(),
                     index=dataset.index)
        df.to_csv(feature_store_path)
