import time

import numpy as np
from sklearn.metrics import roc_auc_score

from src.metrics import MetricsFromScratch

THRESHOLDS = np.linspace(0.05, 0.95, 181)


def find_best_threshold(y_true, y_prob, thresholds=THRESHOLDS):
    """Threshold with the highest F1. Stays at 0.5 if no threshold gives F1 > 0."""
    best_thresh, best_f1 = 0.5, 0.0
    for thresh in thresholds:
        f1 = MetricsFromScratch(y_true, (y_prob >= thresh).astype(int)).f1_score()
        if f1 > best_f1:
            best_thresh, best_f1 = thresh, f1
    return float(best_thresh)


def evaluate_model(model, data):
    """Train, choose the threshold on the validation set, then score once on the test set."""
    start = time.time()
    model.fit(data["X_train"], data["y_train"])
    val_prob = np.asarray(model.predict_proba(data["X_val"]))
    test_prob = np.asarray(model.predict_proba(data["X_test"]))
    elapsed = time.time() - start

    threshold = find_best_threshold(data["y_val"], val_prob)
    val = MetricsFromScratch(data["y_val"], (val_prob >= threshold).astype(int), val_prob)
    test = MetricsFromScratch(data["y_test"], (test_prob >= threshold).astype(int), test_prob)

    return {
        "ROC-AUC": test.roc_auc(),
        "PR-AUC": test.average_precision(),
        "Recall": test.recall(),
        "Precision": test.precision(),
        "F1": test.f1_score(),
        "Accuracy": test.accuracy(),
        "Threshold": threshold,
        "Time (s)": elapsed,
        "val ROC-AUC": val.roc_auc(),
        "test_prob": test_prob,
    }


def sklearn_auc(model, data):
    """Test ROC-AUC of a scikit-learn model trained on the same data, used as a reference."""
    model.fit(data["X_train"], data["y_train"])
    return roc_auc_score(data["y_test"], model.predict_proba(data["X_test"])[:, 1])
