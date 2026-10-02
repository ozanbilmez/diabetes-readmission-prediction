import numpy as np


class VotingClassifierFromScratch:
    """Soft voting: averages the class-1 probabilities of the models."""

    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for _, model in self.models:
            model.fit(X, y)

    def predict_proba(self, X):
        return np.mean([np.asarray(model.predict_proba(X)) for _, model in self.models], axis=0)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
