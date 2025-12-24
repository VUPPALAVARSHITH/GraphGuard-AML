"""
evaluate.py

Evaluation metrics for AML risk prediction.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auc(labels, scores):
    """
    ROC-AUC for transaction-level prediction
    """
    return roc_auc_score(labels, scores)


def precision_recall_at_k(y_true, y_scores, k_percent):
    """
    Precision@K% and Recall@K%
    """
    k = int(len(y_scores) * k_percent / 100)
    top_k_idx = np.argsort(y_scores)[-k:]

    precision = y_true[top_k_idx].sum() / k
    recall = y_true[top_k_idx].sum() / y_true.sum()

    return precision, recall
