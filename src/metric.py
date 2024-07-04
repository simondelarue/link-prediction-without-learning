import numpy as np

from sklearn.metrics import average_precision_score, roc_auc_score

from utils import find_index

    
def compute_auc(labels_true, labels_pred):
    "AUC"
    return roc_auc_score(labels_true, labels_pred)

def compute_ap(labels_true, labels_pred):
    "Average Precision"
    return average_precision_score(labels_true, labels_pred)

def compute_hits_k(labels_true, labels_pred, k):
    """HITS@k"""
    mask_neg = labels_true == 0
    neg_preds = labels_pred[mask_neg]
    sorted_neg_preds = sorted(neg_preds, reverse=True) # Sort values for binary search
    n = len(neg_preds)

    hits = 0
    n_pos = int(np.sum(labels_true))

    for i in range(n_pos):
        pos_pred_val = labels_pred[i]
        rank = find_index(sorted_neg_preds, n, pos_pred_val) + 1 # Binary search
        if rank <= k:
            hits += 1

    return np.sum(hits) / n_pos

def compute_mrr(labels_true, labels_pred):
    """Mean Reciprocal Rank"""
    mask_neg = labels_true == 0
    neg_preds = labels_pred[mask_neg]
    sorted_neg_preds = sorted(neg_preds, reverse=True) # Sort values for binary search
    n = len(neg_preds)

    r_ranks = []
    n_pos = int(np.sum(labels_true))

    for i in range(n_pos):
        pos_pred_val = labels_pred[i]
        rank = find_index(sorted_neg_preds, n, pos_pred_val) + 1 # Binary search
        r_rank = 1 / rank
        r_ranks.append(r_rank)

    return np.mean(r_ranks)
