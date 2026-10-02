import numpy as np


def train_val_test_split(y, groups=None, rng=None, sizes=(0.6, 0.2, 0.2)):
    """Index arrays for the train, validation and test sets.

    With groups (patient IDs) all encounters of a patient end up in the same set,
    so the model is never tested on a patient it has already seen.
    Without groups the split is stratified by class.
    """
    rng = rng if rng is not None else np.random.default_rng()
    y = np.asarray(y)

    if groups is not None:
        groups = np.asarray(groups)
        patients = rng.permutation(np.unique(groups))
        n_train = int(len(patients) * sizes[0])
        n_val = int(len(patients) * sizes[1])
        assigned = [patients[:n_train], patients[n_train:n_train + n_val], patients[n_train + n_val:]]
        return tuple(np.where(np.isin(groups, part))[0] for part in assigned)

    parts = ([], [], [])
    for label in np.unique(y):
        idx = rng.permutation(np.where(y == label)[0])
        n_train = int(len(idx) * sizes[0])
        n_val = int(len(idx) * sizes[1])
        parts[0].append(idx[:n_train])
        parts[1].append(idx[n_train:n_train + n_val])
        parts[2].append(idx[n_train + n_val:])
    return tuple(np.sort(np.concatenate(p)) for p in parts)


class Clipper:
    """Clips outliers to [Q1 - 3*IQR, Q3 + 3*IQR] with quartiles from the training set.

    Binary columns and columns with IQR = 0 (e.g. visit counts that are mostly zero)
    are left alone, otherwise every non-zero value in them would be clipped away.
    """

    def __init__(self, factor=3.0):
        self.factor = factor

    def fit(self, X):
        q1, q3 = np.percentile(X, [25, 75], axis=0)
        iqr = q3 - q1
        n_unique = np.array([len(np.unique(col)) for col in X.T])
        clip = (n_unique > 2) & (iqr > 0)
        self.lower_ = np.where(clip, q1 - self.factor * iqr, -np.inf)
        self.upper_ = np.where(clip, q3 + self.factor * iqr, np.inf)
        return self

    def transform(self, X):
        return np.clip(X, self.lower_, self.upper_)


class Standardizer:
    """Zero mean and unit variance, using the training set's mean and standard deviation."""

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_


def undersample(X, y, rng):
    """Randomly drop rows of the larger class until both classes have the same size."""
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n = min(len(pos), len(neg))
    idx = rng.permutation(np.concatenate([
        rng.choice(pos, n, replace=False),
        rng.choice(neg, n, replace=False),
    ]))
    return X[idx], y[idx]


def select_top_features(X, y, k=35):
    """Indices of the k columns with the highest absolute Pearson correlation with y."""
    correlations = []
    for col in X.T:
        if np.std(col) == 0: correlations.append(0.0)
        else: correlations.append(abs(np.corrcoef(col, y)[0, 1]))
    return np.argsort(correlations)[::-1][:k]


def prepare_data(X, y, groups, rng, top_k=35):
    """Split the data and fit every preprocessing step on the training set only.

    Validation and test keep the real class ratio; only the training set is balanced.
    """
    feature_names = np.asarray(X.columns)
    X = X.to_numpy(dtype=float)
    train, val, test = train_val_test_split(y, groups, rng)

    clipper = Clipper().fit(X[train])
    scaler = Standardizer().fit(clipper.transform(X[train]))

    def transform(idx):
        return scaler.transform(clipper.transform(X[idx]))

    X_train, y_train = undersample(transform(train), y[train], rng)
    cols = select_top_features(X_train, y_train, top_k)

    return {
        "X_train": X_train[:, cols], "y_train": y_train,
        "X_val": transform(val)[:, cols], "y_val": y[val],
        "X_test": transform(test)[:, cols], "y_test": y[test],
        "feature_names": feature_names[cols],
    }
