import numpy as np


class AlwaysNegativeBaseline:
    """Predicts "not readmitted" for every patient. Shows how much accuracy alone is worth."""

    def fit(self, X, y):
        pass

    def predict_proba(self, X):
        return np.zeros(len(X))

    def predict(self, X):
        return np.zeros(len(X), dtype=int)
