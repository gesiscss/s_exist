import os
from itertools import chain, combinations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder

from utils import read_train_no_validation, read_validation, build_feature_path, read_train, read_config

ALL_FEATURE_NAMES = ['sif',
                     'senpai_unclustered_selected',
                     'vader_selected',
                     'male_words_selected',
                     'female_words_selected',
                     'senpai_selected',
                     'most_similar_scale_selected',
                     'perspective_selected',
                     'boosters_selected',
                     'hedges_selected',
                     ]


def find_feature_set_combination():
    train_no_validation = read_train_no_validation()
    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(train_no_validation.task1)
    validation_no_train = read_validation()
    y_validation = y_encoder.fit_transform(validation_no_train.task1)
    all_train_no_validation = read_train()
    y_all_train = y_encoder.fit_transform(all_train_no_validation.task1)

    feature_combo_performances_val = dict()
    feature_combo_performances_cross = dict()

    def powerset(iterable):
        """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
        """
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))

    for feature_combo in powerset(ALL_FEATURE_NAMES):

        train = train_no_validation.copy()
        validation = validation_no_train.copy()
        all_train = all_train_no_validation.copy()
        cols = list()
        for feature in feature_combo:
            if not feature: continue # empty set
            # add to train
            feature_path = build_feature_path('TRAINING_REL', feature)
            feature_df = pd.read_csv(feature_path, index_col='id')
            feature_df.columns = [feature + "_" + column for column in feature_df.columns]
            cols.extend(feature_df.columns)
            train = pd.merge(train, feature_df, how='left', left_index=True, right_index=True)
            # add to validation
            validation = pd.merge(validation, feature_df, how='left', left_index=True, right_index=True)
            # add to all_train
            all_train = pd.merge(all_train, feature_df, how='left', left_index=True, right_index=True)

        # test on validation
        X = train[cols]
        clf = RandomForestClassifier(n_estimators=50, max_depth=5)
        clf.fit(X, y)
        X_validation = validation[cols]
        y_validation_pred = clf.predict(X_validation)
        score = f1_score(y_validation, y_validation_pred)
        print(feature_combo, "%2.4f" % score)
        feature_combo_performances_val[feature_combo] = score

        # test in cross-validation
        X = all_train[cols]
        clf = RandomForestClassifier(n_estimators=50, max_depth=5)
        y_all_train_pred = cross_val_predict(clf, X, y_all_train)
        score = f1_score(y_all_train, y_all_train_pred)
        print(feature_combo, "%2.4f" % score)
        feature_combo_performances_cross[feature_combo] = score
    print('best performance on validation')
    for best_feature_combo, best_performance in sorted(feature_combo_performances_val.items(),
                                                       key=lambda x: x[1], reverse=True)[:10]:
        print(best_feature_combo, best_performance)
    print('best cross-validated performance')
    for best_feature_combo, best_performance in sorted(feature_combo_performances_cross.items(),
                                                       key=lambda x: x[1], reverse=True)[:10]:
        print(best_feature_combo, best_performance)
    performance_df = pd.DataFrame(
        {key: {'validation': feature_combo_performances_val[key],
               'crossvalidation': feature_combo_performances_cross[key],
               }
         for key in feature_combo_performances_val}).T
    save_path = os.path.join(read_config()['DATA_ROOT'], 'feature_set_performance.csv')
    print('saving results to', save_path)
    performance_df.sort_values(by=['validation', ], ascending=False).to_csv(save_path)


def find_model():
    pass
    # automl = autosklearn.classification.AutoSklearnClassifier(
    #     time_left_for_this_task=120,
    #     per_run_time_limit=30,
    #     tmp_folder='/tmp/autosklearn_parallel_1_example_tmp',
    #     output_folder='/tmp/autosklearn_parallel_1_example_out',
    #     n_jobs=4,
    #     # Each one of the 4 jobs is allocated 3GB
    #     memory_limit=3072,
    #     seed=5,
    # )
    # automl.fit(X_train, y_train, dataset_name='breast_cancer')

if __name__ == '__main__':
    find_feature_set_combination()
