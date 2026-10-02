from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import (EXCLUDED_DISCHARGE_IDS, load_diabetes_130, load_hospital_readmissions,
                      map_diagnosis)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DIABETES_CSV = DATA_DIR / "diabetic_data.csv"
HOSPITAL_CSV = DATA_DIR / "hospital_readmissions.csv"


@pytest.fixture(scope="module")
def diabetes():
    return load_diabetes_130(DIABETES_CSV)


@pytest.mark.parametrize("code, group", [
    ("250.83", "Diabetes"), ("428", "Circulatory"), ("486", "Respiratory"),
    ("V57", "Other"), (np.nan, "Other"),
])
def test_map_diagnosis(code, group):
    assert map_diagnosis(code) == group


def test_rows_line_up(diabetes):
    X, y, groups = diabetes
    assert len(X) == len(y) == len(groups)


def test_patients_who_died_or_went_to_hospice_are_removed(diabetes):
    X, _, _ = diabetes
    raw = pd.read_csv(DIABETES_CSV, na_values="?", keep_default_na=False, low_memory=False)
    raw = raw.dropna(subset=["race", "gender", "age", "discharge_disposition_id"])
    assert len(X) == (~raw["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS)).sum()
    assert not any(f"discharge_disposition_id_{i}" in X.columns for i in EXCLUDED_DISCHARGE_IDS)


def test_target_is_readmission_within_30_days(diabetes):
    _, y, _ = diabetes
    assert set(np.unique(y)) == {0, 1}
    assert 0.10 < y.mean() < 0.13


def test_id_columns_are_one_hot_encoded(diabetes):
    X, _, _ = diabetes
    for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        assert col not in X.columns
        assert any(c.startswith(col + "_") for c in X.columns)


def test_all_three_diagnoses_are_used(diabetes):
    X, _, _ = diabetes
    for col in ["diag_1", "diag_2", "diag_3"]:
        assert any(c.startswith(col + "_") for c in X.columns)


def test_unmeasured_lab_results_are_their_own_category(diabetes):
    X, _, _ = diabetes
    # "None" means the test was not done; it must not be read as a missing value
    assert "A1Cresult_None" in X.columns
    assert "max_glu_serum_None" in X.columns


def test_ids_and_target_are_not_features(diabetes):
    X, _, _ = diabetes
    assert not {"patient_nbr", "encounter_id", "readmitted"} & set(X.columns)
    assert X.notna().all().all()
    assert (X.dtypes == float).all()


@pytest.mark.skipif(not HOSPITAL_CSV.exists(), reason="Kaggle file not downloaded (see data/README.md)")
def test_hospital_dataset_loads():
    X, y, groups = load_hospital_readmissions(HOSPITAL_CSV)
    assert groups is None
    assert len(X) == len(y) == 25000
    assert 0.45 < y.mean() < 0.49
    assert "readmitted" not in X.columns
    assert "complexity_index" in X.columns
    assert (X.dtypes == float).all()
