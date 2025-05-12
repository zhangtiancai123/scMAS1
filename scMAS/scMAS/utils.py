from matplotlib import pyplot as plt
from munkres import Munkres
from scipy import sparse
import os
import ctypes
import platform
import scanpy as sc
import pandas as pd
from anndata import AnnData
from scipy.spatial.distance import cdist
import torch
import mkl
import numpy as np
import seaborn as sns  
from scipy.optimize import linear_sum_assignment
from sklearn import metrics

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
mkl.get_max_threads()
C_DIP_FILE = None

def bestMap(L1, L2):
    Label1 = np.unique(L1)
    nClass1 = len(Label1)  
    Label2 = np.unique(L2)
    Label2 = np.where(np.isin(Label1, Label2), Label1, -1)  
    nClass2 = len(Label2)  
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))

    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            G[i, j] = np.sum(ind_cla2 * ind_cla1)

    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]  # 获取匹配索引
    newL2 = np.zeros(L2.shape)

    # 修改后的循环，确保不越界
    for i in range(nClass2):
        # 确保 i 不超过 c 的长度
        if i < len(c):  
            # 匹配结果
            newL2[L2 == Label2[i]] = Label1[c[i]]
        else:
            print(f"警告: Label2 中的标签 {Label2[i]} 没有可匹配的 Label1 标签。")

    return newL2

def Purity_score(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    purity = metrics.accuracy_score(y_true, y_voted_labels)
    return purity


