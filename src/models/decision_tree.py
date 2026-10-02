import numpy as np
from collections import Counter


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None, prob=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.prob = prob  # share of class 1 among the training samples in this leaf

    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeFromScratch:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None, random_state=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features  # features tried at each split: None (all), an int or "sqrt"
        self.random_state = random_state
        self.root = None

    def fit(self, X, y):
        self.n_features_ = self._features_per_split(X.shape[1])
        self.rng = np.random.default_rng(self.random_state)
        self.root = self._grow_tree(X, y)

    def _features_per_split(self, n_total):
        if self.n_features is None: return n_total
        if self.n_features == "sqrt": return max(1, int(np.sqrt(n_total)))
        return min(n_total, self.n_features)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            return self._leaf(y)

        feat_idxs = self.rng.choice(n_feats, self.n_features_, replace=False)
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        if best_feature is None:
            return self._leaf(y)

        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return self._leaf(y)

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(feature=best_feature, threshold=best_thresh, left=left, right=right)

    def _leaf(self, y):
        prob = float(np.mean(y)) if len(y) else 0.0
        return Node(value=self._most_common_label(y), prob=prob)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            # Limit thresholds check for speed on large data
            if len(thresholds) > 100:
                thresholds = np.percentile(X_column, np.linspace(0, 100, 100))

            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thr
        return split_idx, split_thresh

    def _information_gain(self, y, X_column, threshold):
        parent_impurity = self._gini(y)
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0: return 0
        n, n_l, n_r = len(y), len(left_idxs), len(right_idxs)
        e_l, e_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])
        child_impurity = (n_l / n) * e_l + (n_r / n) * e_r
        return parent_impurity - child_impurity

    def _gini(self, y):
        if len(y) == 0: return 0
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum([p**2 for p in proportions])

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        if len(y) == 0: return 0
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._find_leaf(x, self.root).value for x in X])

    def predict_proba(self, X):
        return np.array([self._find_leaf(x, self.root).prob for x in X])

    def _find_leaf(self, x, node):
        if node.is_leaf_node(): return node
        if x[node.feature] <= node.threshold: return self._find_leaf(x, node.left)
        return self._find_leaf(x, node.right)
