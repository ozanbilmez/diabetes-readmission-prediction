import numpy as np


class MetricsFromScratch:
    def __init__(self, y_true, y_pred, y_prob=None):
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.y_prob = np.asarray(y_prob, dtype=float) if y_prob is not None else None

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

    def confusion_matrix(self):
        return np.array([[self.TN, self.FP], [self.FN, self.TP]])

    def _cumulative_counts(self):
        """TP and FP counts with the threshold placed at each distinct score, highest first.

        Equal scores are handled as a single threshold. Otherwise the curve would
        depend on the arbitrary order of tied samples (common with RF and KNN,
        whose probabilities only take a few values).
        """
        order = np.argsort(self.y_prob, kind="mergesort")[::-1]
        scores = self.y_prob[order]
        y = self.y_true[order]
        last_of_each_score = np.r_[np.where(np.diff(scores) != 0)[0], len(y) - 1]
        tps = np.cumsum(y)[last_of_each_score]
        fps = last_of_each_score + 1 - tps
        return tps, fps, scores[last_of_each_score]

    def roc_curve(self):
        tps, fps, thresholds = self._cumulative_counts()
        fpr = np.r_[0, fps / fps[-1]]
        tpr = np.r_[0, tps / tps[-1]]
        return fpr, tpr, np.r_[np.inf, thresholds]

    def roc_auc(self):
        if self.y_prob is None or len(np.unique(self.y_true)) < 2: return 0.0
        fpr, tpr, _ = self.roc_curve()
        # trapezoidal rule
        return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))

    def precision_recall_curve(self):
        tps, fps, thresholds = self._cumulative_counts()
        precision = tps / (tps + fps)
        recall = tps / tps[-1]
        return precision, recall, thresholds

    def average_precision(self):
        """Area under the precision-recall curve as a step sum (same definition as scikit-learn)."""
        if self.y_prob is None or np.sum(self.y_true) == 0: return 0.0
        precision, recall, _ = self.precision_recall_curve()
        return float(np.sum(np.diff(np.r_[0, recall]) * precision))
