from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.svm import LinearSVC

from utils import read_train, read_train_no_validation, read_validation, build_feature_path, read_test


class AbstractTransformer(ABC, TransformerMixin):
    @abstractmethod
    def __init__(self):
        raise NotImplemented('should implement an init setting the transformer attribute')

    def fit(self, X, y=None, **fit_params):
        return self.transformer.fit(X, y, **fit_params)

    def transform(self, X):
        return self.transformer.transform(X)

    @property
    @abstractmethod
    def columns(self):
        raise NotImplemented('should return the column names for the features selected')


class VaderSelector(AbstractTransformer):
    def __init__(self):
        self.transformer = ColumnTransformer(transformers=[('passthrough', 'passthrough', ['compound', 'neu'])],
                                             remainder='drop')

    @property
    def columns(self):
        return ['compound', 'neu']


class OneColumnOneHotEncoder(OneHotEncoder):
    def fit(self, X, y=None, **params):
        return super(OneColumnOneHotEncoder, self).fit(X.values.reshape(-1, 1), y, **params)

    def transform(self, X, **params):
        return super(OneColumnOneHotEncoder, self).transform(X.values.reshape(-1, 1), **params)


class MostSimilarScaleSelector(AbstractTransformer):
    @property
    def columns(self):
        return self.transformer.get_feature_names()

    def __init__(self):
        scale_transformer = OneColumnOneHotEncoder(drop='first', handle_unknown='error')
        sexist_content_transformer = OneColumnOneHotEncoder(drop='first', handle_unknown='error')
        sexist_transformer = OneColumnOneHotEncoder(drop='if_binary', handle_unknown='error')
        self.transformer = ColumnTransformer(transformers=[('sexist', sexist_transformer, 'sexist'),
                                                           ('sexist_content', sexist_content_transformer,
                                                            'sexist_content'),
                                                           ('scale', scale_transformer, 'scale'),
                                                           ])


class AbstractSupportTransformer(AbstractTransformer, ABC):
    @property
    def columns(self):
        return self._columns

    @columns.setter
    def columns(self, value):
        self._columns = value

    def fit(self, X, y, **fit_params):
        super(AbstractSupportTransformer, self).fit(X, y, **fit_params)
        self.columns = np.array(X.columns)[self.transformer.get_support()].tolist()
        return self


class PerspectiveSelector(AbstractSupportTransformer):

    def __init__(self):
        self.transformer = RFECV(ExtraTreesClassifier(n_estimators=5))


class GenderWordsSelector(AbstractSupportTransformer):
    def __init__(self):
        self.transformer = SelectFromModel(LinearSVC(penalty="l1", dual=False))


class SparseSelector(AbstractSupportTransformer):
    def __init__(self):
        self.transformer = SelectFromModel(ExtraTreesClassifier(n_estimators=5))


if __name__ == '__main__':
    train = read_train()
    train_no_validation = read_train_no_validation()
    validation = read_validation()
    test = read_test()

    for feature, transformer in [
        ('vader', VaderSelector()),
        ('perspective', PerspectiveSelector()),
        ('senpai', SparseSelector()),
        ('senpai_unclustered', SparseSelector()),
        ('most_similar_scale', SparseSelector()),
        ('male_words', GenderWordsSelector()),
        ('female_words', GenderWordsSelector()),
        ('hedges', SparseSelector()),
        ('boosters', SparseSelector()),

    ]:
        print('processing', feature)
        in_path = build_feature_path('TRAINING_REL', feature)
        feature_df = pd.read_csv(in_path, index_col='id')
        if feature == 'most_similar_scale':
            trans = MostSimilarScaleSelector()
            transformed = trans.fit_transform(feature_df).todense()
            feature_df = pd.DataFrame(transformed, columns=trans.columns, index=feature_df.index)
        X = feature_df.loc[train_no_validation.index]
        y = train_no_validation.task1
        y_encoder = LabelEncoder()
        y = y_encoder.fit_transform(y)
        print('fit transformer')
        X_t = transformer.fit_transform(X, y)
        selected = transformer.columns
        print('selected', len(selected), 'out of', len(feature_df.columns), ':', selected[:50])
        # score on validation
        clf = RandomForestClassifier(n_estimators=50, max_depth=5)
        clf.fit(X, y)
        y_validation_pred = clf.predict(feature_df.loc[validation.index])
        print('score on validation, pre selection')
        print(classification_report(y_encoder.transform(validation.task1), y_validation_pred))

        clf = RandomForestClassifier(n_estimators=50, max_depth=5)
        clf.fit(X_t, y)
        y_validation_pred = clf.predict(feature_df.loc[validation.index][selected])
        print('score on validation, post selection')
        print(classification_report(y_encoder.transform(validation.task1), y_validation_pred))

        # score in cross-validation
        clf = RandomForestClassifier(n_estimators=50, max_depth=5)
        print('score in cross-validation, pre selection')
        y = y_encoder.transform(train.task1)
        print(classification_report(y, cross_val_predict(estimator=clf,
                                                         X=feature_df.loc[train.index],
                                                         y=y)))

        clf = RandomForestClassifier(n_estimators=50, max_depth=5)
        print('score in cross-validation, post selection')
        y = y_encoder.transform(train.task1)
        print(classification_report(y, cross_val_predict(estimator=clf,
                                                         X=feature_df.loc[train.index, selected],
                                                         y=y)))

        # store
        transformed_df = feature_df.loc[train.index, selected]
        out_path = build_feature_path('TRAINING_REL', feature + '_selected')
        transformed_df.to_csv(out_path)

        in_path = build_feature_path('TEST_REL', feature)
        feature_df = pd.read_csv(in_path, index_col='id')
        if feature == 'most_similar_scale':
            transformed = trans.transform(feature_df).todense()
            feature_df = pd.DataFrame(transformed, columns=trans.columns, index=feature_df.index)
        transformed_df = feature_df.loc[test.index, selected]
        out_path = build_feature_path('TEST_REL', feature + '_selected')
        transformed_df.to_csv(out_path)
