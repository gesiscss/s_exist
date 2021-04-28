from functools import partial

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, ElasticNet
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from preprocessing import preprocess
from utils import read_train, build_feature_path

if __name__ == '__main__':

    print('read training data')
    train = read_train()
    features = ['sif', 'senpai_unclustered', 'senpai']
    for feature in features:
        print('read precomputed feature', feature)
        feature_path = build_feature_path('TRAINING_REL', feature)
        feature_df = pd.read_csv(feature_path, index_col='id')
        feature_df.columns = [feature + "_" + column for column in feature_df.columns]
        train = pd.merge(train, feature_df, how='left', left_index=True, right_index=True)
    print('encode language')
    language_le = LabelEncoder()
    train['language'] = language_le.fit_transform(train.language)
    print('encode platform')
    platform_le = LabelEncoder()
    train['source'] = platform_le.fit_transform(train.source)
    # print('preprocess text')
    # train['text'] = train.text.apply(partial(preprocess, fix_encoding=True))
    features += ['source', 'language', 'text']

    X = train[[column for column in train.columns if any(column.startswith(feature) for feature in features)]]
    y = train.task2.values
    y_le = LabelEncoder()
    y = y_le.fit_transform(y)

    print('X.shape', X.shape, 'y.shape', y.shape, 'unique y', np.unique(y))
    labels = y_le.classes_
    ss = StandardScaler()
    fs = SelectFromModel(estimator=ElasticNet(), max_features=800) #use multitask in case of task2
    clf = Pipeline(steps=[
        ('cv', ColumnTransformer(transformers=[('cv', CountVectorizer(analyzer='char',
                                                                                        ngram_range=(3, 4)), 'text')],
                                                   remainder='passthrough')),
                          # ('ss', ss),
                          # ('fs', fs),
                          ('clf', RandomForestClassifier())])
    skf = StratifiedKFold(n_splits=10)
    y_pred = cross_val_predict(estimator=clf, X=X, y=y, cv=skf)
    print(classification_report(y_true=y, y_pred=y_pred, target_names=labels))

    print('bow baseline')
    y_pred = cross_val_predict(estimator=Pipeline(steps=[('vec', CountVectorizer()), ('clf', MultinomialNB())]),
                               X=train.text, y=y, cv=skf)
    print(classification_report(y_true=y, y_pred=y_pred, target_names=labels))

    print('char ngram baseline')
    y_pred = cross_val_predict(estimator=Pipeline(steps=[('vec', CountVectorizer(analyzer='char', ngram_range=(3, 4))),
                                                         ('clf', MultinomialNB())]),
                               X=train.text, y=y, cv=skf)
    print(classification_report(y_true=y, y_pred=y_pred, target_names=labels))
