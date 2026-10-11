import numpy as np
from sklearn.metrics import average_precision_score, f1_score
FIXED_DECISION_THRESHOLD = 0.5

def average_ranks(scores):
    order = np.argsort(scores, kind='mergesort')
    sorted_scores = scores[order]
    ranks_sorted = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks_sorted[start:end] = 0.5 * (start + 1 + end)
        start = end
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks

def binary_auc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=bool)
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    if positives == 0 or negatives == 0:
        return float('nan')
    ranks = average_ranks(np.asarray(y_score))
    return float((ranks[y_true].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))

def fixed_thresholds(num_classes, threshold=FIXED_DECISION_THRESHOLD):
    num_classes = int(num_classes)
    threshold = float(threshold)
    return np.full(num_classes, threshold, dtype=np.float32)

def classification_metrics(y_true, y_score, thresholds, class_names):
    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    y_pred = y_score >= thresholds[None]
    intersection = np.logical_and(y_true, y_pred).sum(axis=1, dtype=np.float64)
    union = np.logical_or(y_true, y_pred).sum(axis=1, dtype=np.float64)
    true_count = y_true.sum(axis=1, dtype=np.float64)
    predicted_count = y_pred.sum(axis=1, dtype=np.float64)
    sample_accuracy = float(np.mean(intersection / union))
    sample_sensitivity = float(np.mean(intersection / true_count))
    sample_precision_values = np.zeros(len(y_true), dtype=np.float64)
    has_prediction = predicted_count != 0
    sample_precision_values[has_prediction] = intersection[has_prediction] / predicted_count[has_prediction]
    sample_precision_values[np.logical_and(~has_prediction, true_count == 0)] = 1.0
    sample_precision = float(np.mean(sample_precision_values))
    sample_f1 = float(2.0 * sample_sensitivity * sample_precision / (sample_sensitivity + sample_precision)) if sample_sensitivity + sample_precision > 0.0 else 0.0
    auc_values = [binary_auc(y_true[:, index], y_score[:, index]) for index in range(len(class_names))]
    finite_auc = [value for value in auc_values if np.isfinite(value)]
    metrics = {'AUC': float(np.mean(finite_auc)) if finite_auc else float('nan'), 'SEN': sample_sensitivity, 'PRE': sample_precision, 'ACC': sample_accuracy, 'F1': sample_f1, 'mAP': float(average_precision_score(y_true.astype(np.uint8), y_score, average='macro')), 'F1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)), 'F1_micro': float(f1_score(y_true, y_pred, average='micro', zero_division=0))}
    return (metrics, y_pred.astype(np.uint8))
