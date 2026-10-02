import numpy as np


class VotingClassifierFromScratch:
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for name, model in self.models:
            model.fit(X, y)

    def predict_proba(self, X):
        all_probs = []
        for name, model in self.models:
            try:
                p = model.predict_proba(X)
                if isinstance(p, list): p = np.array(p)
                all_probs.append(p)
            except:
                pass 
        
        if not all_probs: return np.zeros(len(X))
        return np.mean(all_probs, axis=0)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.array([1 if p >= 0.5 else 0 for p in probs])
