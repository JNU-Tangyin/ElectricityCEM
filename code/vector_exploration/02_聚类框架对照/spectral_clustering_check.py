# -*- coding: utf-8 -*-
"""
用 k 近邻图和图拉普拉斯特征分解探索谱聚类结果。扫描簇数与邻居数；该方法对参数敏感，
不能预设它比距离方法更稳健。
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_基础"))
import paths
import numpy as np
import pandas as pd
import glob
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler


VECTORS_DIR = paths.CANONICAL
OUTPUT_DIR = os.path.join(paths.exp_dir("exp8_cluster_methods"), "spectral")

K_RANGE = [2, 3, 4]
N_NEIGHBORS_RANGE = [5, 10, 15]
SEED = 42


def load_theta():
    files = sorted(glob.glob(os.path.join(VECTORS_DIR, "*_expert.npy")))
    ids = [os.path.basename(f).replace("_expert.npy", "") for f in files]
    theta = np.array([np.load(f) for f in files])
    return ids, theta


def main():
    ids, theta = load_theta()
    keep = paths.dims_geometry(theta.shape[1])
    x_std = StandardScaler().fit_transform(theta[:, keep])
    print(f">>> 载入 {len(ids)} 户，几何坐标维度 {x_std.shape[1]}")

    for n_neighbors in N_NEIGHBORS_RANGE:
        print(f"\n========== affinity=nearest_neighbors, n_neighbors={n_neighbors} ==========")
        for k in K_RANGE:
            sc = SpectralClustering(
                n_clusters=k,
                affinity="nearest_neighbors",
                n_neighbors=n_neighbors,
                random_state=SEED,
                assign_labels="kmeans",
            )
            labels = sc.fit_predict(x_std)
            sizes = pd.Series(labels).value_counts().sort_index()
            print(f"  K={k}: 各簇大小={dict(sizes)}")
            if sizes.min() <= 10:
                minority_label = sizes.idxmin()
                members = sorted([ids[i] for i in range(len(ids)) if labels[i] == minority_label])
                print(f"    最小簇(n={sizes.min()})成员: {members}")

    out_rows = []
    for n_neighbors in N_NEIGHBORS_RANGE:
        for k in K_RANGE:
            sc = SpectralClustering(
                n_clusters=k, affinity="nearest_neighbors", n_neighbors=n_neighbors,
                random_state=SEED, assign_labels="kmeans",
            )
            labels = sc.fit_predict(x_std)
            for hid, lbl in zip(ids, labels):
                out_rows.append({"n_neighbors": n_neighbors, "k": k, "household_id": hid, "cluster": int(lbl)})
    out_df = pd.DataFrame(out_rows)
    out_path = os.path.join(OUTPUT_DIR, "spectral_clustering_sweep.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\n>>> 已保存: {out_path}")


if __name__ == "__main__":
    main()
