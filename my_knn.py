import numpy as np
from collections import Counter

class KNNFromScratch:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        predictions = [self._predict_one(x) for x in X_test]
        return np.array(predictions)

    def _predict_one(self, x):
        # Euclidean distance
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Get labels
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict_proba(self, X_test):
        probs = []
        for x in X_test:
            distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            # Probability of class 1
            prob_1 = sum(k_nearest_labels) / self.k
            probs.append(prob_1)
        return np.array(probs)