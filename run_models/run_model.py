"""
Provides the code to train and run the simple machine learning models in the `attacks` directory.
"""

import numpy as np
import constants
import scoring_methods

from sklearn.model_selection import StratifiedKFold
from os import path as ospath
from sys import path, exit

dirname, _ = ospath.split(ospath.abspath(__file__))
DATA_DIR = dirname + '/../data/'

# Hack to import from sibling directory
path.append(ospath.dirname(path[0]))

def clean_data(X):
    """
    Goes over a list of features and removes any nan of infinite values

    @param X is a numpy matrix
    """
    for i, row in enumerate(X):
        for j, val in enumerate(row):
            if np.isnan(val) or np.isinf(val):
                X[i][j] = 0

def get_X_y(monitored_data, unmonitored_data):
    """
    Gets a X, y array by combining both the `monitored_data` and `unmonitored_data` lists

    @param monitored_data an array-like matrix that has the following structure `[(features, value)]`
    @param unmonitored_data is also an array-like object: [features]

    @return a tuple `(X, y)` where `X` represents the features and `y` the corresponding labels
    """
    monitored_data = np.array(monitored_data)

    X = list(monitored_data[:, 0])
    X.extend(unmonitored_data)

    y = list(monitored_data[:, 1])
    y.extend([constants.UNMONITORED_LABEL] * len(unmonitored_data))

    X, y = np.array(X), np.array(y)

    clean_data(X)

    return X, y

def k_fold_validation(model, monitored_data, unmonitored_data, k, random_state=123):
    """
    Performs k fold validation on a model. During each fold, records all of the scoring in the `scoring_methods` module.

    @param model is a machine learning model that has the functions `fit(X, y)` and `predict(X)`
    @param monitored_data an array-like matrix that has the following structure `[(features, value)]`
    @param unmonitored_data is also an array-like object: [features]
    @param k is the amount of folds

    @return is a 2D array of scores, with the following structure `[{scoring_method: score}]` where the shape is `len(k)`
    """
    X, y = get_X_y(monitored_data, unmonitored_data)
    skf = StratifiedKFold(n_splits=k, random_state=random_state, shuffle=True)

    evaluations = []
    i = 1
    for train, test in skf.split(X, y):
        print("Starting split {}".format(i))
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        print("Fitting data")
        model.fit(X_train, y_train)

        print("Predicting")
        prediction = model.predict(X_test)

        evaluations.append(scoring_methods.evaluate_model(prediction, y_test))

        print(evaluations[-1])

        i += 1

    return evaluations


def evaluate(model, monitored_data, unmonitored_data, random_state=123):
    """
    You only train on a very small percentage of the unmonitored data

    @param monitored_data an array-like matrix that has the following structure `[(features, value)]`
    @param unmonitored_data is also an array-like object: [features]

    @return a dictionary structured as follows: `{'training_error': [scoring_methods], 'test_error': scoring_methods}`
        where each scoring method is a dict of scoring methods with as key the name and as value the actual score.
    """
    np.random.seed(random_state)

    np.random.shuffle(monitored_data)
    np.random.shuffle(unmonitored_data)

    monitored_split = int(len(monitored_data) * constants.TRAIN_PERCENTAGE_MONITORED)
    unmonitored_split = int(len(unmonitored_data) * constants.TRAIN_PERCENTAGE_UNMONITORED)

    training_evaluations = k_fold_validation(
        model, monitored_data[:monitored_split], unmonitored_data[:unmonitored_split], constants.K_FOLDS, random_state=random_state
    )

    X_train, y_train = get_X_y(monitored_data[:monitored_split], unmonitored_data[:unmonitored_split])
    X_test, y_test = get_X_y(monitored_data[monitored_split:], unmonitored_data[unmonitored_split:])

    print("Calculating test error")
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    test_evaluation = scoring_methods.evaluate_model(prediction, y_test)

    return {
        'training_error': training_evaluations, 'test_error': test_evaluation
    }

def split_data(data):
    """
    Splits the data into a monitored and unmonited set

    @param data is a list of tuples (features, label)
    """
    monitored_data, unmonitored_data = [], []

    for row in data:
        if row[1] == constants.UNMONITORED_LABEL:
            unmonitored_data.append(row[0])

        else:
            monitored_data.append(row)

    return monitored_data, unmonitored_data

def make_data_binary(data):
    """
    Given a data array `(features, label)`, updates the label such that it is either:
    - `constants.UNMONITORED_LABEL`
    - `constant.MONITORED_LABEL`
    """
    for i, row in enumerate(data):
        if row[1] != constants.UNMONITORED_LABEL:
            row = (row[0], constants.MONITORED_LABEL)

def get_models():
    """
    Gets a list of dictionaries where each entry represents a model

    @return `[{'model_name': ..., 'model_constructor': ..., 'path_to_features': ...}]`
    """
    from attacks import kNN1, naive_bayes, random_forest, svc

    return [
        {'model_name': 'kNN', 'model_constructor': kNN1.kNN, 'path_to_features': DATA_DIR + "knn_cells"},
        {'model_name': 'naive_bayes', 'model_constructor': naive_bayes.get_naive_bayes, 'path_to_features': DATA_DIR + "nb_cells"},
        {'model_name': 'random_forest', 'model_constructor': random_forest.get_random_forest, 'path_to_features': DATA_DIR + "rf_cells"},
        {'model_name': 'svc1', 'model_constructor': svc.get_svc, 'path_to_features': DATA_DIR + "svc1_cells"},
        {'model_name': 'svc2', 'model_constructor': svc.get_svc, 'path_to_features': DATA_DIR + "svc2_cells"},
    ]

def get_appropriate_dict(name):
    """
    Given a model name, returns the correct dict
    """
    model_dicts = get_models()

    for val in model_dicts:
        if val['model_name'] == name:
            return val

    return None

def evaluate_model(model_dict, hand_picked_features=True, is_multiclass=True):
    """
    Given a model, performs the an evaluation.

    @param model_dict is a dictionary, as described in `[{'model_name': ..., 'model_constructor': ..., 'path_to_features': ...}]`
    @param hand_picked_features is a boolean value that decides whether to use the hand-picked features or the auto generated once.

    @return a dictionary structured as follows: `{'training_error': [scoring_methods], 'test_error': scoring_methods}`
        where each scoring method is a dict of scoring methods with as key the name and as value the actual score.
    """
    path = model_dict['path_to_features']
    if hand_picked_features:
        path = DATA_DIR + 'af_cells'

    from helpers import pull_data_in_memory

    data = pull_data_in_memory(data_dir=path, extension=".cellf")

    if not is_multiclass:
        make_data_binary(data)

    monitored_data, unmonitored_data = split_data(data)

    model = model_dict['model_constructor'](is_multiclass=is_multiclass)

    return evaluate(model, monitored_data, unmonitored_data)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Runs a specific machine learning model on the appropriate data.")

    parser.add_argument('--model', help="Select which model to run (kNN, random_forest, svc1, scv2)", default="kNN")
    parser.add_argument('--handpicked', action='store_true', help="Whether to use the hand-picked features or automatically generated ones")
    parser.add_argument('--is_multiclass', action='store_true', help="If you are training on a multiclass or binary problem.")

    args = parser.parse_args()

    model_dict = get_appropriate_dict(args.model)
    if model_dict is None:
        print("Model not found. Needs to be one of the following values: (kNN, naive_bayes, random_forest, svc1, scv2)")
        exit(0)

    else:
        res = evaluate_model(model_dict, args.handpicked, args.is_multiclass)
        print(res)
