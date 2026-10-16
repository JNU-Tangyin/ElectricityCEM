# -*- coding: utf-8 -*-
"""
共享 t-SNE / UMAP 脚本的向量读取、特征构造和散点图绘制。
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths



def load_theta(vectors_dir):
    files = sorted(glob.glob(os.path.join(vectors_dir, "*_expert.npy")))
    ids = [os.path.basename(f).replace("_expert.npy", "") for f in files]
    theta = np.array([np.load(f) for f in files])
    return ids, theta


def build_feature_matrix(theta):
    return StandardScaler().fit_transform(theta[:, paths.dims_geometry(theta.shape[1])])


def plot_panel_plain(ax, ids, coords):
    """纯散点 + 户名标注，不做任何分组编码。"""
    ax.scatter(coords[:, 0], coords[:, 1], s=40, alpha=0.8, c="tab:blue", edgecolors="k", linewidths=0.5)
    for i, hid in enumerate(ids):
        ax.annotate(hid, (coords[i, 0], coords[i, 1]), fontsize=6, alpha=0.6)



