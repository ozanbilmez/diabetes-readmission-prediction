import numpy as np

class MetricsFromScratch:
    def __init__(self, y_true, y_pred, y_prob=None):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_prob = np.array(y_prob) if y_prob is not None else None
        
        self.TP = np.sum((self.y_true == 1) & (self.y_pred == 1))
        self.TN = np.sum((self.y_true == 0) & (self.y_pred == 0))
        self.FP = np.sum((self.y_true == 0) & (self.y_pred == 1))
        self.FN = np.sum((self.y_true == 1) & (self.y_pred == 0))

    def accuracy(self):
        return (self.TP + self.TN) / len(self.y_true)

    def precision(self):
        if (self.TP + self.FP) == 0: return 0.0
        return self.TP / (self.TP + self.FP)

    def recall(self):
        if (self.TP + self.FN) == 0: return 0.0
        return self.TP / (self.TP + self.FN)

    def f1_score(self):
        p = self.precision()
        r = self.recall()
        if (p + r) == 0: return 0.0
        return 2 * (p * r) / (p + r)

    def roc_auc(self):
        if self.y_prob is None: return 0.0
        desc_score_indices = np.argsort(self.y_prob)[::-1]
        y_true_sorted = self.y_true[desc_score_indices]
        
        tps = np.cumsum(y_true_sorted)
        fps = np.cumsum(1 - y_true_sorted)
        
        total_pos = np.sum(y_true_sorted)
        total_neg = len(y_true_sorted) - total_pos
        
        if total_pos == 0 or total_neg == 0: return 0.0

        tpr = tps / total_pos
        fpr = fps / total_neg
        
        # Trapezoidal rule for AUC
        tpr = np.insert(tpr, 0, 0)
        fpr = np.insert(fpr, 0, 0)
        return np.trapz(tpr, fpr)