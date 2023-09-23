# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2023-02-03 12:04
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2023-02-03 12:04

import numpy as np
from numpy import ndarray
from math import log
from collections import Counter

def calculate_IW(y: ndarray) -> float:
    # Initialize.
    IW = 0
    if len(y.shape)>1 and max(y)>1:
        xx = np.zeros(y.shape[1])
        part = np.zeros(10)

        # Calculate information entropy.
        for i in range(0, y.shape[0]):
            for j in range(0, y.shape[1]):
                if y[i, j] == 1:
                    xx[j] += 1
        for i in range(0, y.shape[1]):
            part[i] = xx[i] / y.shape[0]
            if part[i] == 0:
                continue
            IW += part[i] * log(part[i], 2) * (-1)
    else:
        for i in range(0, max(y)):
            part = np.sum(y==i) / y.shape[0]
            if part == 0:
                continue
            IW += part * log(part, 2) * (-1)

    # Keep 3 decimals.
    IW = round(IW, 3)

    return IW


def deepcopy_list(list_to_copy: list) -> list:
    if len(list_to_copy) == 0:
        return []
    else:
        if type(list_to_copy[0]) != list:
            return [x for x in list_to_copy]
        else:
            return [deepcopy_list(list_to_copy[x]) for x in range(len(list_to_copy))]


def list_normalization(list_: list) -> list:
    list_min = min(list_)
    list_max = max(list_)
    if list_max == list_min:
        return [1 / len(list_) for i in range(len(list_))]
    else:
        return [(list_[i] - list_min) / (list_max - list_min) for i in range(len(list_))]

def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, np.array(v2).T)
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res

def calculate_SIC(new_label, old_label, c):
    element_new = dict(Counter(new_label))
    element_old = dict(Counter(old_label))
    len_new_label = sum(element_new.values())
    len_old_label = sum(element_old.values())
    label_old=list(element_old.keys())
    label_new=list(element_new.keys())
    label_sum=set(label_old+label_new)
    SIC = 0
    for label in label_sum:
        if label in element_new:
            p_new = element_new[label] / len_new_label
        else:
            p_new = 0
        if label in element_old:
            p_old = element_old[label] / len_old_label
        else:
            p_old = 0
        SIC = SIC + p_new * log((p_new+c) / (p_old+c), 2)
    return SIC

def fedprofkl(label,label_r):
    k_mean=np.mean(label)
    r_mean=np.mean(label_r)
    k_omiga=np.var(label)
    r_omiga=np.var(label_r)
    return 1/2*np.log2(r_omiga/k_omiga)+(k_omiga-r_omiga)/2/r_omiga+(k_mean-r_mean)**2/2/r_omiga