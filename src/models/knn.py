import numpy as np


class KNNFromScratch:
    def __init__(self, k=3, batch_size=500):
        self.k = k
        self.batch_size = batch_size
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)

    def predict_proba(self, X_test):
        X_test = np.asarray(X_test, dtype=float)
        train_sq = np.sum(self.X_train ** 2, axis=1)
        probs = np.empty(len(X_test))
        # Distances are computed for a batch of test rows at once, so memory stays
        # at batch_size x n_train instead of n_test x n_train.
        for start in range(0, len(X_test), self.batch_size):
            batch = X_test[start:start + self.batch_size]
            # squared Euclidean distance: |a|^2 - 2ab + |b|^2 (same ranking as Euclidean)
            dist = np.sum(batch ** 2, axis=1)[:, None] - 2 * batch @ self.X_train.T + train_sq
            nearest = np.argpartition(dist, self.k - 1, axis=1)[:, :self.k]
            probs[start:start + len(batch)] = self.y_train[nearest].mean(axis=1)
        return probs

    def predict(self, X_test):
        return (self.predict_proba(X_test) >= 0.5).astype(int)
