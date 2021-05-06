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
    test['language'] = language_le.fit_transform(test.language)
    print('preprocess text')
    train['text'] = train.text.apply(partial(preprocess, fix_encoding=True))
    test['text'] = test.text.apply(partial(preprocess, fix_encoding=True))
    
    features += ['language', 'text']

    # try on a small sample to see if script works
    # train = train.head(100)

    X = train[[column for column in train.columns if any(column.startswith(feature) for feature in features)]]
    y = train.task2.values
    y_le = LabelEncoder()
    y = y_le.fit_transform(y)

    X_test = test[[column for column in test.columns if any(column.startswith(feature) for feature in features)]]


    # print(X['language'])
    # print(features)

    print('X.shape', X.shape, 'y.shape', y.shape, 'unique y', np.unique(y))
    labels = y_le.classes_
    ss = StandardScaler()
    fs = SelectFromModel(estimator=MultinomialNB())  # use multitask in case of task2

    # add different types of ML models

    names = [
         #"MNB",
         "SVM",
         "LR",
         "RF",
         "SVC"
        ]

    classifiers = [
        #MultinomialNB(),
        LinearSVC(),
        LogisticRegression(),
        RandomForestClassifier(),
        SVC()
        #MLPClassifier()
    ]

    parameters = [
              #{'clf__alpha': (1e-2, 1e-3)},
              {'clf__C': [0.01, 0.1, 1, 10, 100], 'clf__class_weight' : ['balanced', None]},
              {'clf__C': [0.01, 0.1, 1, 10, 100], 'clf__penalty': ['l1', 'l2'], 'clf__class_weight' : ['balanced', None]},
              {'clf__max_depth': (1, 10, 50, 100, 200, 300, 400), 'clf__class_weight' : ['balanced', None]},
              {'clf__gamma': ['scale', 'auto'], 'clf__kernel' : ['rbf', 'poly'], 'clf__class_weight' : ['balanced', None]},
        #     {'clf__alpha': (1e-2, 1e-3)}
             ]
    
    models = {}
    all_results = []

    skf = StratifiedKFold(n_splits=10)
    
    for name, classifier, params in zip(names, classifiers, parameters):
        ppl = Pipeline(steps=[
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
          ('clf', classifier)
          ])
        gs_clf = GridSearchCV(ppl, param_grid=params, n_jobs=-1, cv = skf)
        #clf = gs_clf.fit(X_train, y_train)
        models[name] = gs_clf

        # fit the data 
        gs_clf.fit(X, y)

        print(name)
  
        y_pred = cross_val_predict(estimator=gs_clf.best_estimator_, X=X, y=y, cv=skf)
        print(classification_report(y_true=y, y_pred=y_pred, target_names=labels))

        
        # save the classifier
        with open(MODELPATH + '/exist2021_multiclass_%s.pkl' %(name), 'wb') as fid:
          pickle.dump(gs_clf.best_estimator_, fid)  

        # save results
        cr = classification_report(y_true=y, y_pred=y_pred, target_names=labels, output_dict = True)
        all_results.append(get_results(cr, y, y_pred, labels = dict(zip(range(0, len(train.task2.unique())), sorted(train.task2.unique()))),
         method = name, hyperparams = gs_clf.best_params_))

        # generate test-run
        if True:
            test['predictions'] = gs_clf.best_estimator_.predict(X_test)
            mapping = dict(zip(range(0, len(train.task2.unique())), sorted(train.task2.unique())))
            test['predictions'] = test['predictions'].map(mapping)
            test = test.reset_index()
            test['id'] = test['id'].astype(str)
            test = test[['test_case', 'id', 'predictions']]
            generate_test_run(test, task = "task2", run = "2", model_type = name)
        # except Exception as e:
        #     print(e)
        #     time.sleep(30)
        #     pass


    # result_df = pd.DataFrame(all_results)    
    # print(result_df)  
    # result_df.to_csv("finetuned_classification_result_summary_multiclass.csv", index = False, sep = "\t")    


    #     print('bow baseline')
    #     y_pred = cross_val_predict(estimator=Pipeline(steps=[('vec', CountVectorizer()), ('clf', MultinomialNB())]),
    #                            X=train.text, y=y, cv=skf)
    #     print(classification_report(y_true=y, y_pred=y_pred, target_names=labels))

    #     print('char ngram baseline')
    #     y_pred = cross_val_predict(estimator=Pipeline(steps=[('vec', CountVectorizer(analyzer='char', ngram_range=(3, 4))),
    #                                                      ('clf', MultinomialNB())]),
    #                            X=train.text, y=y, cv=skf)
    #     print(classification_report(y_true=y, y_pred=y_pred, target_names=labels))
    # 
