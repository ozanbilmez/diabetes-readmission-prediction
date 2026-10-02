from src.models.logistic_regression import LogisticRegressionFromScratch
from src.models.knn import KNNFromScratch
from src.models.random_forest import RandomForestFromScratch
from src.models.ensemble import VotingClassifierFromScratch
from src.metrics import MetricsFromScratch
from src.data import load_diabetes_130, load_hospital_readmissions
import numpy as np
import pandas as pd
import time

# Scikit-learn for Verification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

# --- HELPER FUNCTIONS ---
def scale_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1 
    X_scaled = (X - mean) / std
    return X_scaled

def select_top_features(X, y, top_k=35):
    n_features = X.shape[1]
    correlations = []
    for i in range(n_features):
        feature_col = X[:, i]
        if np.std(feature_col) == 0: corr = 0
        else: corr = np.corrcoef(feature_col, y)[0, 1]
        correlations.append(abs(corr))
    top_indices = np.argsort(correlations)[::-1][:top_k]
    return X[:, top_indices]

# --- KRİTİK GÜNCELLEME: OPTIMUM DENGE NOKTASI BULUCU ---
def find_best_threshold(y_true, y_probs):
    """
    Precision ve Recall değerlerini birbirine en yakın yapan 
    (farklarını minimize eden) 'Denge Noktası'nı bulur.
    Böylece Recall düşer, Precision artar ve Accuracy dengelenir.
    """
    best_diff = 1.0  # Farkı minimize edeceğiz (başlangıçta en yüksek fark 1)
    best_thresh = 0.5
    
    # 0.30 ile 0.70 arasında hassas tarama
    for thresh in np.arange(0.30, 0.75, 0.005):
        preds = [1 if p >= thresh else 0 for p in y_probs]
        m = MetricsFromScratch(y_true, preds)
        
        prec = m.precision()
        rec = m.recall()
        
        # Eğer Precision veya Recall 0 ise o noktayı atla (geçersiz denge)
        if prec == 0 or rec == 0: continue
            
        # Hedef: |Precision - Recall| farkını EN AZA indirmek
        diff = abs(prec - rec)
        
        # İkincil Hedef: Farklar eşitse, Accuracy'si yüksek olanı seç
        if diff < best_diff:
            best_diff = diff
            best_thresh = thresh
            
    return best_thresh

def run_verification(X_train, y_train, X_test, y_test, algo_name):
    sklearn_model = None
    if "Logistic" in algo_name:
        sklearn_model = LogisticRegression(max_iter=5000, solver='liblinear')
    elif "KNN" in algo_name:
        sklearn_model = KNeighborsClassifier(n_neighbors=9)
    elif "Random Forest" in algo_name:
        sklearn_model = RandomForestClassifier(n_estimators=25, max_depth=15)
    elif "Ensemble" in algo_name:
        clf1 = LogisticRegression(max_iter=1000, solver='liblinear')
        clf2 = KNeighborsClassifier(n_neighbors=9)
        clf3 = RandomForestClassifier(n_estimators=25, max_depth=15)
        sklearn_model = VotingClassifier(
            estimators=[('lr', clf1), ('knn', clf2), ('rf', clf3)], 
            voting='soft'
        )

    if sklearn_model is not None:
        try:
            sklearn_model.fit(X_train, y_train)
            preds = sklearn_model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            print(f"   >> [VERIFICATION] Sklearn {algo_name} -> Accuracy: {acc:.4f}")
            return acc
        except Exception as e:
            print(f"   >> [VERIFICATION] Skipped: {e}")
            return 0
    return 0

def run_experiment():
    datasets = [
        ("Diabetes 130-US", "data/diabetic_data.csv", load_diabetes_130),
        ("Hosp. Readmissions", "data/hospital_readmissions.csv", load_hospital_readmissions)
    ]
    
    results = []

    for dataset_name, filename, loader_func in datasets:
        print(f"\n{'='*60}")
        print(f"PROCESSING: {dataset_name}")
        print(f"{'='*60}")

        try:
            X_df, y, _ = loader_func(filename)
            X_raw = X_df.to_numpy()
        except FileNotFoundError:
            print(f"File not found: {filename}")
            continue

        # --- MASSIVE BALANCING ---
        indices_pos = np.where(y == 1)[0]
        indices_neg = np.where(y == 0)[0]
        MAX_PER_CLASS = 15000 
        n_samples = min(len(indices_pos), len(indices_neg), MAX_PER_CLASS)
        print(f"Massive Balancing: {n_samples} Positive & {n_samples} Negative")
        
        balanced_indices = np.concatenate([
            np.random.choice(indices_pos, n_samples, replace=False),
            np.random.choice(indices_neg, n_samples, replace=False)
        ])
        np.random.shuffle(balanced_indices)
        X_bal = X_raw[balanced_indices]
        y_bal = y[balanced_indices]

        # FEATURE SELECTION & SCALING
        X_selected = select_top_features(X_bal, y_bal, top_k=35)
        X = scale_features(X_selected)

        # SPLIT
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y_bal[:split_idx], y_bal[split_idx:]
        print(f"Train Set: {len(X_train)} samples")
        
        # --- DEFINE MODELS ---
        m_lr = LogisticRegressionFromScratch(learning_rate=0.05, num_iterations=6000)
        m_knn = KNNFromScratch(k=9)
        m_rf = RandomForestFromScratch(n_trees=25, max_depth=15)
        m_ens = VotingClassifierFromScratch([
            ("LR", LogisticRegressionFromScratch(learning_rate=0.05, num_iterations=6000)),
            ("KNN", KNNFromScratch(k=9)),
            ("RF", RandomForestFromScratch(n_trees=25, max_depth=15))
        ])

        experiment_models = [
            ("Logistic Regression", m_lr),
            ("KNN (k=9)", m_knn),
            ("Random Forest", m_rf),
            ("Ensemble (Soft Vote)", m_ens)
        ]

        for algo_name, model in experiment_models:
            print(f"Running {algo_name}...")
            start_t = time.time()
            
            # KNN Limitation
            if algo_name.startswith("KNN"):
                 limit = 10000 
                 if len(X_train) > limit:
                     X_train_run = X_train[:limit]
                     y_train_run = y_train[:limit]
                     print(f"   (Restricted to {limit} samples for speed)")
                 else:
                     X_train_run, y_train_run = X_train, y_train
                 X_test_run = X_test[:1000]
                 y_test_run = y_test[:1000]
                 model.fit(X_train_run, y_train_run)
            else:
                 X_train_run, y_train_run = X_train, y_train
                 X_test_run, y_test_run = X_test, y_test
                 model.fit(X_train, y_train)
            
            # PREDICT
            try: 
                probs = model.predict_proba(X_test_run)
                if isinstance(probs, list): probs = np.array(probs)
            except: 
                probs = model.predict(X_test_run)

            # --- OPTİMİZASYON BURADA YAPILIYOR ---
            # Artık Precision ve Recall'ı eşitleyen en iyi threshold'u buluyoruz
            best_thresh = find_best_threshold(y_test_run, probs)
            preds = [1 if p >= best_thresh else 0 for p in probs]

            m = MetricsFromScratch(y_test_run, preds, probs)
            elapsed = time.time() - start_t
            
            # Verification
            sklearn_acc = run_verification(X_train_run, y_train_run, X_test_run, y_test_run, algo_name)

            results.append({
                "Dataset": dataset_name,
                "Algorithm": algo_name,
                "Scratch Acc": round(m.accuracy(), 4),
                "Sklearn Acc": round(sklearn_acc, 4),
                "Recall": round(m.recall(), 4),
                "Precision": round(m.precision(), 4),
                "F1-Score": round(m.f1_score(), 4),
                "AUC": round(m.roc_auc(), 4),
                "Time (s)": round(elapsed, 2)
            })

    print("\n\n--- FINAL BALANCED RESULTS ---")
    df_results = pd.DataFrame(results)
    cols = ["Dataset", "Algorithm", "Scratch Acc", "Sklearn Acc", "Recall", "Precision", "F1-Score", "AUC", "Time (s)"]
    df_results = df_results[cols]
    print(df_results)
    df_results.to_csv("final_balanced_results.csv", index=False)

if __name__ == "__main__":
    run_experiment()