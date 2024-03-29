import math

import torch
import numpy as np


def precision(ground_truth_indexes, k):
    return (ground_truth_indexes < k).to(float).sum() / k


def recall(ground_truth_indexes, k):
    return (ground_truth_indexes < k).to(float).sum() / ground_truth_indexes.size(0)


# def f1score(ground_truth_indexes, k):
#     count = 0
#     for i in ground_truth_indexes:
#         if i >= k:
#             break
#         count += 1
#     p = count / k
#     r = count / len(ground_truth_indexes)
#     return 2 * p * r / (p + r) if (p + r) > 0.0 else 0.0


# Pre-compute ideal DCGs for performance improvement
IDEAL_DCG = np.zeros((1000,))
IDEAL_DCG[0] = 0
for _i in range(1, 1000):
    IDEAL_DCG[_i] = IDEAL_DCG[_i-1] + 1/math.log2(_i+1)


def nDCG(ground_truth_indexes, k):
    ground_truth_indexes = ground_truth_indexes.to(float)
    dcg = 1 / torch.log2(ground_truth_indexes + 2)
    dcg.scatter_(0, (ground_truth_indexes >= k).nonzero(as_tuple=True)[0], 0)
    dcg = dcg.sum()
    return dcg / IDEAL_DCG[(ground_truth_indexes < k).sum()] if dcg > 0 else 0


# def average_precision(ground_truth_indexes, k):
#     tot = 0
#     hits = 0
#     for i in ground_truth_indexes:
#         if i >= k:
#             break
#         hits += 1
#         tot += hits / (i+1)
#     return tot / k


def auc_exact(ground_truth_indexes, inventory_size):
    n = ground_truth_indexes.size(0)
    assert inventory_size >= n
    if inventory_size == n:
        return 1
    i = torch.arange(1, n + 1, device=ground_truth_indexes.device)
    idx = ground_truth_indexes + 1
    auc = (((inventory_size - idx) - (n - i))).sum(dtype=torch.float64)
    auc /= (inventory_size - n)
    auc /= n
    return auc


def reciprocal_rank(ground_truth_indexes):
    return 1 / (ground_truth_indexes.min().to(float) + 1)


# def jaccard_index(set1, set2):
#     x = sum(1 for e in set1 if e in set2)
#     return x / (len(set1) + len(set2) - x)
