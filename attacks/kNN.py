"""
Attack based on the following paper "Effective Attacks and Provable Defenses for Website Fingerprinting" by T. Wang et al.
"""

from sys import stdout

class kNN():
    """
    k Neirest Neighbour classifier described in the paper outlined above.

    Essentially it calculates the distance of different instances by using a set of weights described below:
    $$ d(P, P') = \sum_{1 \leq i \leq |F|} w_i |f_i(P) - f_i(P')| $$

    Attributes:
        - is_multiclass does not make a difference. It is just part of the interface used for models
        - K_CLOSEST_NEIGHBORS is the amount of neighbours it uses for a majority vote
        - weights is a list of `AMOUNT_FEATURES` length that is used to signify how 'important' features are
        - K_RECO is the number of closest neighbours used for weight learning
    """

    def __init__(self, is_multiclass=True, K_CLOSEST_NEIGHBORS=5):
        # Constants
        self.K_RECO = 5.0 # Num of neighbors for weight learning

        self.K_CLOSEST_NEIGHBORS = K_CLOSEST_NEIGHBORS

    def _init_weights(self):
        """
        Randomly assign the weights a value between 0.5 and 1.5
        """
        from random import uniform

        self.weights = [uniform(0.5, 1.5) for _ in range(self.AMOUNT_FEATURES)]

    def _calculate_dist(self, point1, point2):
        """
        Calculates the distance between 2 points as follows:
        $$ d(P, P') = \sum_{1 \leq i \leq |F|} w_i |f_i(P) - f_i(P')| $$

        @param point1, point2 are two lists, with each item in the list being a feature.
        """
        if isinstance(point1, dict):
            point1 = point1['point']

        if isinstance(point2, dict):
            point2 = point2['point']

        dist = 0
        for i, f1 in enumerate(point1):
            f2 = point2[i]
            w = self.weights[i]

            # TODO: Check why this step is needed
            if f1 != -1 and f2 != -1:
                dist += w * abs(f1 - f2)

        return dist

    def _calculate_all_dist(self, point, data, index=-1):
        """
        Calculates the distances between the point and all other data points.

        @param the index of the point in data (-1 if not defined) *(So we don't include it since it will be 0 anyway)*.

        @return a list of distances
        """
        distances = []
        for i, row in enumerate(data):
            if i == index:
                distances.append(float("inf"))

            else:
                dist = self._calculate_dist(point, row)
                distances.append(dist)

        return distances

    def _calculate_all_dist_for_feature(self, point, data, feature_index, index=-1):
        """
        Calculates the distance to all other points for a specific feature

        @param point is the point from where we are calculating the distance
        @param data
        @param feature_index is the index of the feature we are examining
        @param index is the index of the point. So we can ignore it (since the distance would be 0 anyway)

        @return a list of distances
        """
        distances = []
        for i, row in enumerate(data):
            if i == index:
                distances.append(float("inf"))

            else:
                dist = self.weights[index] * abs(row['point'][feature_index] - point[feature_index])
                distances.append(dist)

        return distances

    def _find_closest_reco_points(self, label, data):
        """
        Find the `K_RECO`-closest members that either have the same label and the ones that have another

        @param label is the label of the point we are examining
        @param data is a list as follows: [{'point': data, 'distance': distance, 'label': label}]

        @return a tuple of (same_label, different_label) where each a list of length K_RECO
        """
        # Sort on distance
        data = sorted(data, key=lambda x: x['distance'])

        same_label, different_label = [], []

        for val in data:
            if len(same_label) == self.K_RECO and len(different_label) == self.K_RECO:
                break

            if val['label'] == label and len(same_label) < self.K_RECO:
                same_label.append(val)

            elif val['label'] != label and len(different_label) < self.K_RECO:
                different_label.append(val)

        return (same_label, different_label)

    def _get_point_baddness(self, same_label, different_label):
        """
        Calculates the fraction of different label points that are closer than the maximum
        in the `same_label` list.

        @param same_label is a list of the `K_RECO` closest points to the current point we are examining with the same label as that point
        @param different_label is a list of the `K_RECO` closest points to the current point we are examining with a different label as that point

        @return a measure of how bad the feature is
        """
        max_good = max(same_label, key=lambda x: x['distance'])
        point_badness = len([x for x in different_label if x['distance'] < max_good['distance']])

        point_badness /= self.K_RECO # Calculate fraction

        return point_badness

    def _update_weights(self, point_badness, features_badness):
        """
        The features_badness gives us a measure of how bad certain features are.

        For every `i, w in enumerate(self.weights)` (except for the weight with the minimum feature badness), we decrease the weight by `0.01 * w`
        Then we increase all the weights by `min(features_badness)`

        See paper for extra steps

        @param point_badness is the general measure of how bad a point is classified
        @param features_badness is a measure of how bad a measure is
        """
        min_badness = min(features_badness)

        # Make all weights smaller
        for i, w in enumerate(self.weights):

            # Skip the minimum
            if features_badness[i] == min_badness:
                continue

            subtract = w * 0.01 * (features_badness[i] / self.K_RECO) * (0.2 + (point_badness / self.K_RECO))

            self.weights[i] -= subtract

        # Increase weights to maintain $d(P_{train}, S_{bad})$
        for i, w in enumerate(self.weights):
            self.weights[i] += min_badness

    def _learn_weights(self, data, labels):
        """
        In the training process, learns a set of weights

        @param data is a 2D matrix with the data samples and features
        @param labels is a 1D list where `class_of(data[i]) == labels[i]` for all i in `range(len(data))`
        """
        training = min(len(data), 2000)

        for i in range(training):
            update_progess(training, i)

            row = data[i]
            distances = self._calculate_all_dist(row, data, index=i)
            new_data = [{'point': data[k], 'distance': distances[k], 'label': labels[k]} for k in range(len(data))]

            same_label, different_label = self._find_closest_reco_points(labels[i], new_data)

            features_badness = []

            # Go over all features
            for j, feature in enumerate(row):
                # import pdb; pdb.set_trace()
                # print(different_label)
                diff_label_distances = self._calculate_all_dist_for_feature(row, different_label, j)
                same_label_distances = self._calculate_all_dist_for_feature(row, same_label, j)

                diff_label_distances = [{'distance': x} for x in diff_label_distances]
                same_label_distances = [{'distance': x} for x in same_label_distances]

                point_badness_feature = self._get_point_baddness(same_label_distances, diff_label_distances)

                features_badness.append(point_badness_feature)

            point_badness = self._get_point_baddness(same_label, different_label)
            self._update_weights(point_badness, features_badness)

    def _majority_vote(self, points):
        votes = {}

        for point in points:
            if point['label'] not in votes:
                votes[point['label']] = 0
            votes[point['label']] += 1

        return max(votes, key=votes.get)

    def fit(self, X, y):
        """
        Trains the model
        """
        self.AMOUNT_FEATURES = len(X[0])
        self._init_weights()

        self._learn_weights(X, y)

        self.data = [{'point': X[i], 'label': y[i]} for i in range(len(X))]


    def predict(self, X):
        """
        Predicts using a majority vote

        @param X is an array-like object
        """
        if self.data is None or len(self.data) == 0:
            raise Exception("Train model first!")

        elif len(X) == 0:
            raise Exception("Cannot predict on empty array")

        elif len(X[0]) != self.AMOUNT_FEATURES:
            raise Exception("Does not match the shape with {} features".format(self.AMOUNT_FEATURES))

        else:
            predicted = []
            for i, point in enumerate(X):
                update_progess(len(X), i)
                dists = self._calculate_all_dist(point, self.data)

                points = [{'distance': dists[i], 'label': self.data[i]['label']} for i in range(len(self.data))]
                points = sorted(points, key=lambda x: x['distance'])[:self.K_CLOSEST_NEIGHBORS]

                predicted.append(self._majority_vote(points))

            return predicted

def update_progess(total, current):
    """Prints a percentage of how far the process currently is"""
    stdout.write("{:.2f} %\r".format((current/total) * 100))
    stdout.flush()

# Tests that have nothing to do with website fingerprinting
if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    from functools import reduce

    data = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=0.1)

    classifier = kNN()

    classifier.fit(X_train, y_train)
    prediction = classifier.predict(X_test)

    print("ACTUAL   : {}".format(list(y_test)))
    print("PREDICTED: {}".format(prediction))

    accuracy = reduce(lambda acc, x: acc + 1 if x[0] == x[1] else acc, zip(list(y_test), prediction), 0)
    accuracy /= len(prediction)
    accuracy *= 100

    print("Achieved a {} % accuracy".format(accuracy))
