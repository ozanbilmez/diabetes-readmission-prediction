import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.evaluation import evaluate_model, find_best_threshold, sklearn_auc
from src.models import AlwaysNegativeBaseline, LogisticRegressionFromScratch


def toy_data(seed=0):
    rng = np.random.default_rng(seed)

    def make(n, positive_rate):
        y = (rng.random(n) < positive_rate).astype(int)
        X = np.column_stack([y + rng.normal(scale=0.5, size=n), rng.normal(size=n)])
        return X, y

    X_train, y_train = make(400, 0.5)
    X_val, y_val = make(300, 0.2)
    X_test, y_test = make(300, 0.2)
    return {"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test}


def test_best_threshold_maximises_f1():
    y = np.array([0, 0, 0, 1, 1, 1])
    prob = np.array([0.1, 0.2, 0.32, 0.58, 0.7, 0.8])
    assert 0.32 < find_best_threshold(y, prob) <= 0.58


def test_threshold_falls_back_to_half_when_nothing_is_predicted_positive():
    assert find_best_threshold(np.array([0, 1, 0, 1]), np.zeros(4)) == 0.5


def test_evaluate_model_reports_test_metrics():
    result = evaluate_model(LogisticRegressionFromScratch(num_iterations=500), toy_data())
    assert result["ROC-AUC"] > 0.8
    assert 0 < result["Threshold"] < 1
    assert len(result["test_prob"]) == 300
    assert {"PR-AUC", "Recall", "Precision", "F1", "Accuracy", "val ROC-AUC", "Time (s)"} <= result.keys()


def test_threshold_is_chosen_on_validation_not_test():
    data = toy_data()
    model = LogisticRegressionFromScratch(num_iterations=500)
    result = evaluate_model(model, data)
    assert result["Threshold"] == find_best_threshold(data["y_val"], model.predict_proba(data["X_val"]))


def test_baseline_scores_like_always_saying_no():
    data = toy_data()
    result = evaluate_model(AlwaysNegativeBaseline(), data)
    prevalence = data["y_test"].mean()
    assert result["ROC-AUC"] == pytest.approx(0.5)
    assert result["PR-AUC"] == pytest.approx(prevalence)
    assert result["Accuracy"] == pytest.approx(1 - prevalence)
    assert result["Recall"] == 0.0
    assert result["Threshold"] == 0.5


def test_sklearn_auc_is_measured_on_test_set():
    assert sklearn_auc(LogisticRegression(), toy_data()) > 0.8
