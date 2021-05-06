from functools import partial

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB

from feature_selection import VIFSelector
from preprocessing import preprocess
from utils import read_train, read_test, build_feature_path, \
read_modelpath, get_results, generate_test_run

import pickle

import time

MODELPATH = read_modelpath()


if __name__ == '__main__':

    print('read training and test data')
    train = read_train()
    test = read_test()
    features = ['sif',
                'senpai_unclustered_selected',
                'vader_selected',
                'male_words_selected',
                'female_words_selected',
                'senpai_selected',
                'most_similar_scale_selected',
                'perspective_selected',
                'perspective_difference_selected', # file doesn't exist
                'boosters_selected',
                'hedges_selected',
                # 'mentions_selected',
                'hashtags_selected', # file doesn't exist
                'mentions_total',
                # 'hashtags_total',
                # 'mentions_count',
                # 'hashtags_count',
                'bert_binary', # need to encode
                'bert_multiclass', # need to encode
                'bert_avg_all_but_first_binary_scaled',
                'bert_avg_all_but_first_multiclass_scaled', 
                ]
    for data, data_type in [(train, "TRAINING_REL"), (test, "TEST_REL")]:
        for feature in features:       
            print('read precomputed feature', feature)
            feature_path = build_feature_path(data_type, feature)
            feature_df = pd.read_csv(feature_path, index_col='id')
            feature_df.columns = [feature + "_" + column for column in feature_df.columns]
            # encode the bert features, use the labels in the training data
            if feature == 'bert_binary' or feature == 'bert_multiclass':
                label_mapping = {"bert_binary" : {'sexist' : 1, 'non-sexist' : 0},
                           "bert_multiclass" : dict(zip(sorted(train.task2.unique()),
                            range(0, len(train.task2.unique()))))
                            }

                feature_df['%s_predictions' %(feature)] = feature_df['%s_predictions' %(feature)]\
                  .map(label_mapping[feature])
            data = pd.merge(data, feature_df, how='left', left_index=True, right_index=True)

    print('encode language')
    language_le = LabelEncoder()
    train['language'] = language_le.fit_transform(train.language)
    test['language'] = language_le.transform(test.language)
    print('preprocess text')
    train['text'] = train.text.apply(partial(preprocess, fix_encoding=True))
    test['text'] = test.text.apply(partial(preprocess, fix_encoding=True))
    
    features += ['language', 'text']

    # try on a small sample to see if script works
    # train = train.head(100)

    X = train[[column for column in train.columns if any(column.startswith(feature) for feature in features)]]
    y = train.task1.values
    y_le = LabelEncoder()
    y = y_le.fit_transform(y)

    X_test = test[[column for column in test.columns if any(column.startswith(feature) for feature in features)]]


    # print(X['language'])
    # print(features)

    print('X.shape', X.shape, 'y.shape', y.shape, 'unique y', np.unique(y))
    labels = y_le.classes_
    ss = StandardScaler(with_mean=False)
    fs = SelectFromModel(estimator=MultinomialNB())  # use multitask in case of task2

    clf = Pipeline(steps=[
        ('cv', ColumnTransformer(transformers=[('cv', Pipeline(steps=[('cv', CountVectorizer(analyzer='char',
                                                                      max_df=.5,
                                                                      min_df=.1,
                                                                      ngram_range=(3, 4))),
                                                                      ('fs', fs)
                                                                      ]), 'text')],
                                 remainder='passthrough'
                                 )),
        #                   ('ss', ss),
        # ('fs', fs),
        ('clf', RandomForestClassifier())])

    test = test.reset_index()
    test['id'] = test['id'].astype(str)

    skf = StratifiedKFold(n_splits=10)
    y_pred = cross_val_predict(estimator=clf, X=X, y=y, cv=skf)
    print(classification_report(y_true=y, y_pred=y_pred, target_names=labels))
    clf.fit(X, y)
    test['predictions'] = clf.predict(X_test)
    test['predictions'] = test['predictions'].map({1 : 'sexist', 0 : 'non-sexist'})
    test = test[['test_case', 'id', 'predictions']]
    generate_test_run(test, task = "task1", run = "2", model_type = "RF_untuned")

    print('bow baseline')
    clf = Pipeline(steps=[('vec', CountVectorizer()), ('clf', MultinomialNB())])
    y_pred = cross_val_predict(estimator=clf,
                               X=train.text, y=y, cv=skf)
    print(classification_report(y_true=y, y_pred=y_pred, target_names=labels))
    clf.fit(X.text, y)
    test['predictions'] = clf.predict(X_test.text)
    test['predictions'] = test['predictions'].map({1 : 'sexist', 0 : 'non-sexist'})
    test = test[['test_case', 'id', 'predictions']]
    generate_test_run(test, task = "task1", run = "2", model_type = "bow")

    print('char ngram baseline')
    clf = Pipeline(steps=[('vec', CountVectorizer(analyzer='char', ngram_range=(3, 4))),
                                                         ('clf', MultinomialNB())])
    y_pred = cross_val_predict(estimator=clf,
                               X=train.text, y=y, cv=skf)
    print(classification_report(y_true=y, y_pred=y_pred, target_names=labels))
    clf.fit(X.text, y)
    test['predictions'] = clf.predict(X_test.text)
    test['predictions'] = test['predictions'].map({1 : 'sexist', 0 : 'non-sexist'})
    test = test[['test_case', 'id', 'predictions']]
    generate_test_run(test, task = "task1", run = "2", model_type = "char_ngram")

    