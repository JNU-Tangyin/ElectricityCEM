# -*- coding: utf-8 -*-
"""
用 HDBSCAN 按密度可达性探索分组，扫描最小簇规模。结果依赖所用向量与参数，
与其他方法在同一数据上的比较不构成独立验证。
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_基础"))
import paths
import numpy as np
import pandas as pd
import glob
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler


VECTORS_DIR = paths.CANONICAL


MIN_CLUSTER_SIZES = [3, 5, 7, 10]


def load_theta():
    files = sorted(glob.glob(os.path.join(VECTORS_DIR, "*_expert.npy")))
    ids = [os.path.basename(f).replace("_expert.npy", "") for f in files]
    theta = np.array([np.load(f) for f in files])
    return ids, theta


def main():
    ids, theta = load_theta()
    print(f">>> 载入 {len(ids)} 户，theta维度 {theta.shape[1]}")

    keep_idx = paths.dims_geometry(theta.shape[1])
    x_std = StandardScaler().fit_transform(theta[:, keep_idx])
    print(f"\n========== 几何坐标 ({len(keep_idx)} 维) ==========")
    for mcs in MIN_CLUSTER_SIZES:
        hdb = HDBSCAN(min_cluster_size=mcs, metric="euclidean")
        labels = hdb.fit_predict(x_std)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        sizes = pd.Series(labels).value_counts().sort_index()

        print(f"  min_cluster_size={mcs}: 簇数={n_clusters}, 噪声点数={n_noise}, 各簇大小={dict(sizes)}")

        if n_clusters >= 1:
            hdb_df = pd.DataFrame({"household_id": ids, "hdbscan_label": labels})
            for lbl in sorted(set(labels)):
                if lbl == -1:
                    continue
                members = sorted(hdb_df.loc[hdb_df["hdbscan_label"] == lbl, "household_id"])
                print(f"    簇{lbl}(n={len(members)}): {members}")

            out_path = os.path.join(
                paths.exp_dir("exp8_cluster_methods"), "hdbscan", f"hdbscan_mcs{mcs}.csv")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            hdb_df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
