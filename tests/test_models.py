import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from src.models import (AlwaysNegativeBaseline, DecisionTreeFromScratch, KNNFromScratch,
                        LogisticRegressionFromScratch, RandomForestFromScratch,
                        VotingClassifierFromScratch)


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(n_samples=600, n_features=10, n_informative=5, random_state=0)
    return X[:450], y[:450], X[450:], y[450:]


@pytest.fixture(scope="module")
def separable():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 10))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X[:300], y[:300], X[300:], y[300:]


def test_knn_probabilities_match_sklearn(data):
    X_train, y_train, X_test, _ = data
    ours = KNNFromScratch(k=9)
    ours.fit(X_train, y_train)
    theirs = KNeighborsClassifier(n_neighbors=9).fit(X_train, y_train)
    np.testing.assert_allclose(ours.predict_proba(X_test), theirs.predict_proba(X_test)[:, 1])


def test_knn_result_does_not_depend_on_batch_size(data):
    X_train, y_train, X_test, _ = data
    small, large = KNNFromScratch(k=5, batch_size=7), KNNFromScratch(k=5, batch_size=1000)
    small.fit(X_train, y_train)
    large.fit(X_train, y_train)
    np.testing.assert_array_equal(small.predict_proba(X_test), large.predict_proba(X_test))


def test_logistic_regression_auc_is_close_to_sklearn(data):
    X_train, y_train, X_test, y_test = data
    ours = LogisticRegressionFromScratch(learning_rate=0.05, num_iterations=3000)
    ours.fit(X_train, y_train)
    theirs = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    ours_auc = roc_auc_score(y_test, ours.predict_proba(X_test))
    theirs_auc = roc_auc_score(y_test, theirs.predict_proba(X_test)[:, 1])
    assert abs(ours_auc - theirs_auc) < 0.01


@pytest.mark.parametrize("make_model", [
    lambda: DecisionTreeFromScratch(max_depth=8, random_state=0),
    lambda: RandomForestFromScratch(n_trees=10, max_depth=8, random_state=0),
])
def test_tree_models_learn_a_clear_boundary(separable, make_model):
    X_train, y_train, X_test, y_test = separable
    model = make_model()
    model.fit(X_train, y_train)
    assert np.mean(model.predict(X_test) == y_test) > 0.85


def test_random_forest_is_reproducible(separable):
    X_train, y_train, X_test, _ = separable
    a = RandomForestFromScratch(n_trees=5, max_depth=5, random_state=42)
    b = RandomForestFromScratch(n_trees=5, max_depth=5, random_state=42)
    a.fit(X_train, y_train)
    b.fit(X_train, y_train)
    np.testing.assert_array_equal(a.predict_proba(X_test), b.predict_proba(X_test))


def test_random_forest_uses_sqrt_features_per_split_by_default(separable):
    X_train, y_train, _, _ = separable
    forest = RandomForestFromScratch(n_trees=2, max_depth=2, random_state=0)
    forest.fit(X_train, y_train)
    assert [tree.n_features_ for tree in forest.trees] == [3, 3]


def test_tree_probabilities_are_leaf_class_ratios():
    X = np.array([[0.0], [0.0], [0.0], [1.0]])
    y = np.array([0, 1, 1, 0])
    tree = DecisionTreeFromScratch(max_depth=1, random_state=0)
    tree.fit(X, y)
    np.testing.assert_allclose(tree.predict_proba(np.array([[0.0], [1.0]])), [2 / 3, 0.0])


def test_ensemble_averages_member_probabilities(data):
    X_train, y_train, X_test, _ = data
    members = [("LR", LogisticRegressionFromScratch(num_iterations=200)), ("KNN", KNNFromScratch(k=5))]
    ensemble = VotingClassifierFromScratch(members)
    ensemble.fit(X_train, y_train)
    expected = np.mean([model.predict_proba(X_test) for _, model in members], axis=0)
    np.testing.assert_allclose(ensemble.predict_proba(X_test), expected)


def test_ensemble_does_not_hide_member_errors(data):
    class Broken:
        def fit(self, X, y):
            pass

        def predict_proba(self, X):
            raise RuntimeError("broken model")

    ensemble = VotingClassifierFromScratch([("broken", Broken())])
    ensemble.fit(data[0], data[1])
    with pytest.raises(RuntimeError):
        ensemble.predict_proba(data[2])


def test_baseline_never_predicts_readmission(data):
    X_train, y_train, X_test, _ = data
    model = AlwaysNegativeBaseline()
    model.fit(X_train, y_train)
    assert not model.predict(X_test).any()
    assert not model.predict_proba(X_test).any()
