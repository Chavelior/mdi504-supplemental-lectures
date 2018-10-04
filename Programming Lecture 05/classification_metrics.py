import numpy as np

def precision(C_model, C_truth):
    TP = np.sum(np.logical_and(C_model == 1, C_truth == 1))
    FP = np.sum(np.logical_and(C_model == 1, C_truth == 1))
    
    return TP / (TP + FP)

def recall(C_model, C_truth):
    TP = np.sum(np.logical_and(C_model == 1, C_truth == 1))
    FN = np.sum(np.logical_and(C_model == 0, C_truth == 1))
    
    return TP / (TP + FN)


def F_beta(C_model, C_truth, beta=1):
    return (beta+1)/(beta/precision(C_model, C_truth) + \1/recall(C_model, C_truth))