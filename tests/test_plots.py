import numpy as np
import pandas as pd

from src import plots


def test_all_figures_are_written(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    probs = {"Model A": rng.random(200), "Model B": np.clip(0.3 * y + 0.7 * rng.random(200), 0, 1)}
    table = pd.DataFrame({
        "Dataset": ["A", "A", "B"],
        "Model": ["LR", "Baseline", "LR"],
        "ROC-AUC": [0.65, 0.5, 0.6],
        "sklearn ROC-AUC": [0.651, np.nan, 0.6],
    })

    plots.plot_roc_pr(y, probs, "Toy data", tmp_path / "roc_pr.png")
    plots.plot_scratch_vs_sklearn(table, tmp_path / "compare.png")
    plots.plot_feature_importance(rng.normal(size=20), np.array([f"f{i}" for i in range(20)]), tmp_path / "fi.png")
    plots.plot_confusion_matrix(np.array([[50, 10], [5, 35]]), "Toy data", tmp_path / "cm.png")

    for name in ["roc_pr.png", "compare.png", "fi.png", "cm.png"]:
        assert (tmp_path / name).stat().st_size > 10_000
