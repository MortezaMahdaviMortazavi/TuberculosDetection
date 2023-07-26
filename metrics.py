import torch

def true_positive(y_true,y_pred):
    return ((y_true == 1) & (y_pred == 1)).sum().item()

def true_negative(y_true,y_pred):
    return ((y_true == 0) & (y_pred == 0)).sum().item()

def false_positive(y_true,y_pred):
    return ((y_true == 0) & (y_pred == 1)).sum().item()

def false_negative(y_true,y_pred):
    return ((y_true == 1) & (y_pred == 0)).sum().item()


def precision(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    return tp / (tp + fp + 1e-7)

def recall(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    return tp / (tp + fn + 1e-7) 

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec + 1e-7)

def accuracy(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    total_samples = y_true.shape[0]
    return (tp + tn) / total_samples
