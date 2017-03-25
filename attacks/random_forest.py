"""
k-fingerprinting attack.
First trains a random forest on the data.
Then using the training data, it extracts a set of fingerprints (which are the ID's of all of the leaves that were 'activated').

Finally, if we want to classify a new instance, we essentially extract the fingerprint using the random forest.
Next, we use a kNN technique to find the k closest neighbours (using the Hamming distance).
If all of those neighbours have the same class, we classify the instance as that class.
If not, we classify it as a not previously seen class.

For now we assume that the data fits in memory.
"""

from sklearn.ensemble import RandomForestClassifier

class kFingerprinting:
    """
    The k Fingerprinting model

    Attributes:
        forest - Random forest classifier
        fingerprints - An array of tuples, that for each training instance contains: (fingerprint, prediction)
        k_neighbours - Amount of closest neighbours the model needs to match before it is classified as the appropriate instance
        is_multiclass - Depending on the thread model, we can either have a multiclass problem or a binary, this boolean flag changes that
        unmonitored_label - The label of the unmonitored websites
    """
    def __init__(self, num_trees=20, k_neighbours=3, is_multiclass=True, unmonitored_label=-1):
        self.forest = RandomForestClassifier(n_jobs=2, n_estimators=num_trees, oob_score=True)
        self.k_neighbours = k_neighbours
        self.is_multiclass = is_multiclass
        self.fingerprints = []

    def fit(self, X, y):
        self.forest.fit(X, y)
        self.fingerprints = list(zip(forest.apply(X), y))

    def get_closest_neighbours(self, x):
        """
        Finds the `k_neighbours` closest neightbour to the fingerprint x

        @param x is a fingerprint of a newly predicted instance
        @return an array of the k closest neighbours in the format (distance, label)
        """
        distances = []
        for fp in self.fingerprints:
            dist = len(list(filter(lambda elem: elem[0] != elem[1], zip(x, fp[0]))))
            distances.append((dist, fp[1])) # (Distance, label)

        distances.sort()
        return distances[:self.k_neighbours]

    def predict(self, X):
        """
        Given a list of X instances, it extracts the fingerprints and classifies it as an instance
        or as -1 if none of the classes match (unseen instance).

        If the `is_multiclass` is set to false, the class is either 1 or -1 (1 if it belongs to the set of seen websites)
        """
        new_fingerprints = self.forest.apply(X)
        res = []
        for elem in new_fingerprints:
            neighbours = self.get_closest_neighbours(elem)

            # Checks if the `k_neighbours` closest neightbours have the same label
            neighbours_match = all(x[1] == neighbours[0][1] for x in neighbours)

            if neighbours_match:
                res.append(neighbours[0][1])
            else:
                res.append(unmonitored_label) # Classify as unseen website

        return res

def get_random_forest(num_trees=20, k_neighbours=3, is_multiclass=True, unmonitored_label=-1):
    return kFingerprinting(num_trees=num_trees, k_neighbours=k_neighbours, is_multiclass=is_multiclass, unmonitored_label=unmonitored_label)
