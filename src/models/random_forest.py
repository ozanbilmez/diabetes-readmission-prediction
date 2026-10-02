import numpy as np
from collections import Counter
from src.models.decision_tree import DecisionTreeFromScratch

class RandomForestFromScratch:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            # Decision Tree Oluştur
            tree = DecisionTreeFromScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            
            # Bootstrap Sampling (Veriden rastgele örneklem al)
            # Bu sayede her ağaç verinin farklı bir kısmını görerek uzmanlaşır.
            n_samples = X.shape[0]
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[idxs], y[idxs]
            
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Tüm ağaçlardan tahminleri topla
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # Oylama Yap (Majority Vote)
        # Sütun bazında (her bir örnek için) en çok tekrar eden tahmini bul
        final_preds = []
        for i in range(X.shape[0]):
            votes = tree_preds[:, i]
            counter = Counter(votes)
            most_common = counter.most_common(1)[0][0]
            final_preds.append(most_common)
            
        return np.array(final_preds)

    def predict_proba(self, X):
        # Olasılık hesabı: Ağaçların kaçı "1" dedi?
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Ortalama alarak olasılık bul (Örn: 10 ağacın 8'i '1' dediyse olasılık 0.8)
        probs = np.mean(tree_preds, axis=0)
        return probs