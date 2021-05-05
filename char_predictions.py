from functools import partial

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import StratifiedKFold, cross_val_predict, HalvingGridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from preprocessing import preprocess
from utils import read_train, build_feature_path, read_test

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':
    print('read training and test data')
    train = read_train()
    test = read_test()
    print('preprocess text')
    train['text'] = train.text.apply(partial(preprocess, fix_encoding=True))
    test['text'] = test.text.apply(partial(preprocess, fix_encoding=True))

    X = train['text']
    X_test = test['text']

    y = train.task1.values
    y_le = LabelEncoder()
    y = y_le.fit_transform(y)

    labels = y_le.classes_

    clf = Pipeline(steps=[('cv', CountVectorizer(analyzer='char')),
                          ('fs', SelectFromModel(estimator=MultinomialNB())),
                          ('clf', RandomForestClassifier())])

    param_grid = {'cv__lowercase': [True],
                  'cv__ngram_range': [(2, 5), (3, 4)],
                  'cv__max_df': [1., .5],
                  'cv__min_df': [0., .1],
                  'cv__max_features': [None, 2000],
                  'cv__binary': [True],
                  'fs__threshold': ['median', 1e-4, -np.inf],
                  'clf__n_estimators': [100],
                  'clf__max_depth': [5, 7],
                  'clf__min_samples_leaf': [1, 50],
                  }
    print('fitting')
    gscv = HalvingGridSearchCV(estimator=clf, param_grid=param_grid, refit=True, verbose=1, cv=3, n_jobs=-1)
    gscv.fit(X, y)
    print('predict on test')
    y_test_pred = gscv.predict(X_test)
    y_test_df = pd.Series(y_test_pred, index=test.index)

    print('predict on train')
    clf.set_params(**gscv.best_params_)
    skf = StratifiedKFold(n_splits=10)
    y_train_pred = cross_val_predict(estimator=clf, X=X, y=y, cv=skf)
    y_train_df = pd.Series(y_train_pred, index=train.index)

    for dataset, dataset_key in [(y_train_df, 'TRAINING_REL'),
                                 (y_test_df, 'TEST_REL')]:
        print('saving', dataset_key)
        feature_store_path = build_feature_path(dataset_key, 'char_prediction')
        dataset.to_csv(feature_store_path)
