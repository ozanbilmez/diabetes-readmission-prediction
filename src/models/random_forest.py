import numpy as np
from src.models.decision_tree import DecisionTreeFromScratch


class RandomForestFromScratch:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features="sqrt", random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        self.trees = []
        n_samples = X.shape[0]
        for _ in range(self.n_trees):
            tree = DecisionTreeFromScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features,
                random_state=rng.integers(2**32),
            )

            # Bootstrap sample: every tree sees a different resample of the data,
            # and with a random feature subset at each split the trees disagree more,
            # which is what makes averaging them useful.
            idxs = rng.choice(n_samples, n_samples, replace=True)
            tree.fit(X[idxs], y[idxs])
            self.trees.append(tree)

    def predict_proba(self, X):
        # Average of the class-1 share in the leaf each tree ends up in
        return np.mean([tree.predict_proba(X) for tree in self.trees], axis=0)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
