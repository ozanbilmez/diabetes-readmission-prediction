"""Runs every model on both datasets and writes the results to results/."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from src.data import load_diabetes_130, load_hospital_readmissions
from src.evaluation import evaluate_model, sklearn_auc
from src.models import (AlwaysNegativeBaseline, KNNFromScratch, LogisticRegressionFromScratch,
                        RandomForestFromScratch, VotingClassifierFromScratch)
from src.preprocessing import prepare_data

SEED = 42
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

DATASETS = [
    ("diabetes", "Diabetes 130-US", DATA_DIR / "diabetic_data.csv", load_diabetes_130),
    ("hospital", "Hospital Readmissions", DATA_DIR / "hospital_readmissions.csv", load_hospital_readmissions),
]

COLUMNS = ["Dataset", "Model", "ROC-AUC", "sklearn ROC-AUC", "PR-AUC", "Recall", "Precision",
           "F1", "Accuracy", "Threshold", "Time (s)"]

BASELINE = "Baseline (always no)"
LR_PARAMS = dict(learning_rate=0.05, num_iterations=6000)
KNN_K = 9
RF_PARAMS = dict(n_trees=25, max_depth=15, n_features="sqrt", random_state=SEED)


def build_models():
    """(name, from-scratch model, scikit-learn counterpart or None)"""
    def sklearn_rf():
        return RandomForestClassifier(n_estimators=25, max_depth=15, max_features="sqrt", random_state=SEED)

    return [
        (BASELINE, AlwaysNegativeBaseline(), None),
        ("Logistic Regression", LogisticRegressionFromScratch(**LR_PARAMS), LogisticRegression(max_iter=1000)),
        ("KNN (k=9)", KNNFromScratch(k=KNN_K), KNeighborsClassifier(n_neighbors=KNN_K)),
        ("Random Forest", RandomForestFromScratch(**RF_PARAMS), sklearn_rf()),
        ("Ensemble (soft vote)",
         VotingClassifierFromScratch([
             ("LR", LogisticRegressionFromScratch(**LR_PARAMS)),
             ("KNN", KNNFromScratch(k=KNN_K)),
             ("RF", RandomForestFromScratch(**RF_PARAMS)),
         ]),
         VotingClassifier([
             ("lr", LogisticRegression(max_iter=1000)),
             ("knn", KNeighborsClassifier(n_neighbors=KNN_K)),
             ("rf", sklearn_rf()),
         ], voting="soft")),
    ]


def run_dataset(name, path, loader):
    print(f"\n=== {name} ===")
    X, y, groups = loader(path)
    data = prepare_data(X, y, groups, rng=np.random.default_rng(SEED))
    print(f"train {len(data['y_train'])} (balanced) | val {len(data['y_val'])} | "
          f"test {len(data['y_test'])} | readmitted in test: {data['y_test'].mean():.1%}")

    rows, results, models = [], {}, {}
    for model_name, model, sklearn_model in build_models():
        print(f"  {model_name:<22}", end="", flush=True)
        result = evaluate_model(model, data)
        result["sklearn ROC-AUC"] = sklearn_auc(sklearn_model, data) if sklearn_model is not None else np.nan
        print(f"ROC-AUC {result['ROC-AUC']:.3f}  sklearn {result['sklearn ROC-AUC']:.3f}  "
              f"{result['Time (s)']:.0f}s")
        rows.append({"Dataset": name, "Model": model_name, **result})
        results[model_name] = result
        models[model_name] = model
    return rows, data, results, models


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    for key, name, path, loader in DATASETS:
        if not path.exists():
            print(f"\nSkipping {name}: {path.name} not found (see data/README.md)")
            continue
        dataset_rows, data, results, models = run_dataset(name, path, loader)
        rows += dataset_rows

    table = pd.DataFrame(rows)[COLUMNS].round(4)
    table.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    print("\n" + table.to_string(index=False))


if __name__ == "__main__":
    main()
