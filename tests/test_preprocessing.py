import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (Clipper, Standardizer, prepare_data, select_top_features,
                               train_val_test_split, undersample)


def test_group_split_keeps_each_patient_in_one_set():
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 300, 1000)
    y = rng.integers(0, 2, 1000)
    train, val, test = train_val_test_split(y, groups, rng)
    patients = [set(groups[idx]) for idx in (train, val, test)]
    assert not patients[0] & patients[1]
    assert not patients[0] & patients[2]
    assert not patients[1] & patients[2]
    assert len(train) + len(val) + len(test) == 1000


def test_stratified_split_keeps_class_ratio():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.11).astype(int)
    train, val, test = train_val_test_split(y, None, rng)
    for idx in (train, val, test):
        assert y[idx].mean() == pytest.approx(y.mean(), abs=0.01)
    assert len(np.unique(np.concatenate([train, val, test]))) == 5000
    assert len(train) == pytest.approx(3000, abs=2)


def test_standardizer_uses_training_statistics_only():
    train = np.array([[1.0, 5.0], [3.0, 5.0]])
    scaler = Standardizer().fit(train)
    np.testing.assert_allclose(scaler.mean_, [2.0, 5.0])
    # constant column: std 0 is replaced by 1 instead of dividing by zero
    np.testing.assert_allclose(scaler.transform(np.array([[100.0, 7.0]])), [[98.0, 2.0]])


def test_clipper_bounds_come_from_training_data():
    clipper = Clipper().fit(np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]))
    # Q1 = 2, Q3 = 4, IQR = 2 -> bounds [-4, 10]
    clipped = clipper.transform(np.array([[50.0], [-50.0], [3.0]]))
    np.testing.assert_allclose(clipped, [[10.0], [-4.0], [3.0]])


def test_clipper_leaves_binary_and_zero_iqr_columns_alone():
    # column 0: one-hot dummy; column 1: mostly zeros like number_emergency (IQR = 0)
    train = np.array([[0, 0], [1, 0], [0, 0], [0, 0], [1, 0], [0, 0], [1, 0], [0, 0], [1, 2], [0, 9]],
                     dtype=float)
    clipper = Clipper().fit(train)
    np.testing.assert_array_equal(clipper.transform(train), train)


def test_undersample_balances_classes_and_keeps_rows_with_labels():
    rng = np.random.default_rng(0)
    y = np.array([0] * 90 + [1] * 10)
    X = np.arange(100, dtype=float).reshape(-1, 1)
    X_bal, y_bal = undersample(X, y, rng)
    assert (y_bal == 1).sum() == (y_bal == 0).sum() == 10
    assert set(X_bal[y_bal == 1, 0]) == set(range(90, 100))


def test_select_top_features_picks_most_correlated_columns():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    X = np.column_stack([
        rng.normal(size=500),                   # unrelated
        y + rng.normal(scale=0.1, size=500),    # strong
        np.ones(500),                           # constant
        y + rng.normal(scale=1.0, size=500),    # weaker
    ])
    np.testing.assert_array_equal(select_top_features(X, y, k=2), [1, 3])


def test_prepare_data_balances_train_and_keeps_test_realistic():
    rng = np.random.default_rng(0)
    n = 3000
    y = (rng.random(n) < 0.15).astype(int)
    X = pd.DataFrame({
        "signal": y + rng.normal(size=n),
        "noise": rng.normal(size=n),
        "flag": rng.integers(0, 2, n).astype(float),
    })
    data = prepare_data(X, y, groups=np.arange(n) // 2, rng=rng, top_k=2)
    assert data["y_train"].mean() == 0.5
    assert data["y_test"].mean() == pytest.approx(0.15, abs=0.03)
    assert data["X_train"].shape[1] == data["X_val"].shape[1] == data["X_test"].shape[1] == 2
    assert "signal" in data["feature_names"]
    assert len(data["X_val"]) == len(data["y_val"])
