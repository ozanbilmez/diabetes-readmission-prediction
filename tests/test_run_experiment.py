import pandas as pd

import run_experiment


def test_metrics_file_has_no_timing_column():
    # timings change on every run, so they would make the committed results differ each time
    assert "Time (s)" not in run_experiment.COLUMNS


def test_skipped_dataset_keeps_its_previous_rows(tmp_path):
    previous = pd.DataFrame({"Dataset": ["Diabetes 130-US", "Hospital Readmissions"],
                             "Model": ["LR", "LR"], "ROC-AUC": [0.60, 0.63]})
    previous.to_csv(tmp_path / "metrics.csv", index=False)
    new = pd.DataFrame({"Dataset": ["Diabetes 130-US"], "Model": ["LR"], "ROC-AUC": [0.67]})

    table = run_experiment.keep_previous_rows(new, tmp_path / "metrics.csv", skipped=["Hospital Readmissions"])

    assert table["Dataset"].tolist() == ["Diabetes 130-US", "Hospital Readmissions"]
    assert table["ROC-AUC"].tolist() == [0.67, 0.63]


def test_nothing_is_kept_without_a_previous_file(tmp_path):
    new = pd.DataFrame({"Dataset": ["Diabetes 130-US"], "Model": ["LR"], "ROC-AUC": [0.67]})
    table = run_experiment.keep_previous_rows(new, tmp_path / "missing.csv", skipped=["Hospital Readmissions"])
    assert table.equals(new)
