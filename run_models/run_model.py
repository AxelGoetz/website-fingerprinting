"""
Provides the code to train and run the simple machine learning models in the `attacks` directory.
"""

import numpy as np
import constants
import scoring

from sklearn.model_selection import StratifiedKFold
from os import path as ospath
from sys import path

dirname, _ = ospath.split(ospath.abspath(__file__))
DATA_DIR = dirname + '/../data/

# Hack to import from sibling directory
path.append(ospath.dirname(path[0]))

def get_X_y(monitored_data, unmonitored_data):
    """
    Gets a X, y array by combining both the `monitored_data` and `unmonitored_data` lists

    @param monitored_data an array-like matrix that has the following structure `[(features, value)]`
    @param unmonitored_data is also an array-like object: [features]

    @return a tuple `(X, y)` where `X` represents the features and `y` the corresponding labels
    """
    monitored_data = np.array(monitored_data)

    X = monitored_data[:, 0]
    X.extend(unmonitored_data)

    y = monitored_data[:, 1]
    y.extend([constants.UNMONITORED_LABEL] * len(unmonitored_data))

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

    for train, test in skf.split(X, y):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

        evaluations.append(scoring.evaluate_model(prediction, y_test))

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
        model, monitored_data[:monitored_split], unmonitored_data[:unmonitored_split], constants.K_FOLDS, random_state=random_state=123
    )

    X_train, y_train = get_X_y(monitored_data[:monitored_split], unmonitored_data[:unmonitored_split])
    X_test, y_test = get_X_y(monitored_data[monitored_split:], unmonitored_data[unmonitored_split])

    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    test_evaluation = scoring.evaluate_model(prediction, y_test)

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
    for row in data:
        if row[1] != constants.UNMONITORED_LABEL:
            row[1] = constants.MONITORED_LABEL

def get_models():
    """
    Gets a list of dictionaries where each entry represents a model

    @return `[{'model_name': ..., 'model_constructor': ..., 'path_to_features': ...}]`
    """
    import attacks

    return [
        {'model_name': 'kNN', 'model_constructor': attacks.kNN.kNN, 'path_to_features': DATA_DIR + "knn_cells"},
        {'model_name': 'naive_bayes', 'model_constructor': attacks.naive_bayes.get_naive_bayes, 'path_to_features': DATA_DIR + "nb_cells"},
        {'model_name': 'random_forest', 'model_constructor': attacks.random_forest.get_random_forest, 'path_to_features': DATA_DIR + "rf_cells"},
        {'model_name': 'svc1', 'model_constructor': attacks.svc.get_svc, 'path_to_features': DATA_DIR + "svc1_cells"},
        {'model_name': 'svc2', 'model_constructor': attacks.svc.get_svc, 'path_to_features': DATA_DIR + "svc2_cells"},
    ]

def evaluate_model(model_dict, hand_picked_features=True, is_multiclass=True):
    """
    Given a model, performs the an evaluation.

    @param model_dict is a dictionary, as described in `[{'model_name': ..., 'model_constructor': ..., 'path_to_features': ...}]`
    @param hand_picked_features is a boolean value that decides whether to use the hand-picked features or the auto generated once.

    @return a dictionary structured as follows: `{'training_error': [scoring_methods], 'test_error': scoring_methods}`
        where each scoring method is a dict of scoring methods with as key the name and as value the actual score.
    """
    path = model_dict['path_to_features']
    if not hand_picked_features:
        path = DATA_DIR + 'af_cells'

    from helpers import pull_data_in_memory

    data = pull_data_in_memory(data_dir=DATA_DIR, in_memory=True, extension=".cellf")

    if not is_multiclass:
        make_data_binary(data)

    monitored_data, unmonitored_data = split_data()

    model = model_dict['model_constructor'](is_multiclass=is_multiclass)

    return evaluate(model, monitored_data, unmonitored_data)
