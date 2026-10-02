import numpy as np
import pytest
from sklearn import metrics as skm

from src.metrics import MetricsFromScratch


@pytest.fixture
def scores():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    # rounded scores give many ties, like random forest or KNN probabilities
    prob = np.round(np.clip(0.3 * y + rng.random(500) * 0.7, 0, 1) * 9) / 9
    return y, (prob >= 0.5).astype(int), prob


def test_threshold_metrics_match_sklearn(scores):
    y, pred, prob = scores
    m = MetricsFromScratch(y, pred, prob)
    assert m.accuracy() == pytest.approx(skm.accuracy_score(y, pred))
    assert m.precision() == pytest.approx(skm.precision_score(y, pred))
    assert m.recall() == pytest.approx(skm.recall_score(y, pred))
    assert m.f1_score() == pytest.approx(skm.f1_score(y, pred))


def test_confusion_matrix_matches_sklearn(scores):
    y, pred, _ = scores
    np.testing.assert_array_equal(MetricsFromScratch(y, pred).confusion_matrix(), skm.confusion_matrix(y, pred))


def test_roc_auc_handles_tied_scores(scores):
    y, pred, prob = scores
    assert MetricsFromScratch(y, pred, prob).roc_auc() == pytest.approx(skm.roc_auc_score(y, prob))


def test_roc_curve_matches_sklearn(scores):
    y, pred, prob = scores
    fpr, tpr, _ = MetricsFromScratch(y, pred, prob).roc_curve()
    sk_fpr, sk_tpr, _ = skm.roc_curve(y, prob, drop_intermediate=False)
    np.testing.assert_allclose(fpr, sk_fpr)
    np.testing.assert_allclose(tpr, sk_tpr)


def test_average_precision_matches_sklearn(scores):
    y, pred, prob = scores
    assert MetricsFromScratch(y, pred, prob).average_precision() == pytest.approx(skm.average_precision_score(y, prob))


def test_roc_auc_of_constant_score_is_half():
    y = np.array([0, 1, 0, 1, 1])
    prob = np.zeros(5)
    m = MetricsFromScratch(y, prob, prob)
    assert m.roc_auc() == pytest.approx(0.5)
    assert m.average_precision() == pytest.approx(0.6)


def test_no_positive_predictions_give_zero_not_an_error():
    m = MetricsFromScratch(np.array([0, 1, 1]), np.zeros(3, dtype=int))
    assert m.precision() == 0.0
    assert m.f1_score() == 0.0
