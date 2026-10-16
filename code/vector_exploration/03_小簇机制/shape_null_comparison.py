# -*- coding: utf-8 -*-
"""
将观测点云及 HDBSCAN 小簇特征与高斯零参照比较。簇由同一数据选出，因此本脚本只用于
检查探索到的模式对零参照的敏感性，不构成独立验证。
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_基础"))
import paths
import argparse
import glob
import itertools
import warnings

import numpy as np

import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

MIN_CLUSTER_SIZE = 4
MIN_SAMPLES = 2
N_REP = 300


def load_panel():
    seeds = sorted(d for d in os.listdir(paths.SEED_ROOT)
                   if glob.glob(os.path.join(paths.SEED_ROOT, d, "*_expert.npy")))
    per = {}
    for s in seeds:
        files = sorted(glob.glob(os.path.join(paths.SEED_ROOT, s, "*_expert.npy")))
        ids = [os.path.basename(f).replace("_expert.npy", "") for f in files]
        per[s] = (ids, np.array([np.load(f) for f in files]))
    ids = per[seeds[0]][0]
    keep = paths.dims_geometry(per[seeds[0]][1].shape[1])
    scaler = StandardScaler().fit(np.vstack([per[s][1][:, keep] for s in seeds]))
    return ids, seeds, {s: scaler.transform(per[s][1][:, keep]) for s in seeds}


def describe_minority(M):
    """在一份点云上跑 HDBSCAN，返回少数簇的规模、密度间隙比、成员方向余弦。"""
    lab = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES).fit_predict(M)
    real = [c for c in set(lab) if c != -1]
    if not real:
        return None
    sm = min(real, key=lambda c: (lab == c).sum())
    mem = np.where(lab == sm)[0]
    if len(mem) < 3:
        return None
    oth = np.setdiff1d(np.arange(len(M)), mem)
    D = squareform(pdist(M))
    np.fill_diagonal(D, np.inf)
    nn_in = np.mean([D[a, mem[mem != a]].min() for a in mem])
    nn_out = np.mean([D[a, oth].min() for a in mem]) if len(oth) else np.nan
    V = M - M.mean(axis=0)
    U = V / np.linalg.norm(V, axis=1, keepdims=True)
    cos_mem = np.mean([U[a] @ U[b] for a, b in itertools.combinations(mem, 2)])
    return {"size": len(mem), "gap_ratio": nn_out / nn_in, "cos_members": cos_mem}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_rep", type=int, default=N_REP)
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()
    out_dir = args.output_dir or os.path.join(
        paths.SHAPE, "exp20_shape_null_comparison")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    ids, seeds, X = load_panel()
    n, d = len(ids), X[seeds[0]].shape[1]
    print(f">>> {n} 户 × {len(seeds)} 重训 × {d} 维，零参照重复 {args.n_rep} 次\n")

    # ---------- (1) 云是否被拉长 ----------
    print("=" * 78)
    print(">>> (1) 点云是否存在真实的优势方向（PC1 占比 vs 各向同性零参照）")
    print("=" * 78)
    obs_pc1 = {s: PCA().fit(X[s]).explained_variance_ratio_[0] for s in seeds}
    null_pc1 = np.array([PCA().fit(rng.standard_normal((n, d))).explained_variance_ratio_[0]
                         for _ in range(args.n_rep)])
    print(f"  各向同性零参照的 PC1 占比: 均值 {null_pc1.mean():.3f}，"
          f"95% 分位 {np.quantile(null_pc1, 0.95):.3f}（朴素参照 1/{d}={1/d:.3f}）")
    for s in seeds:
        p = (1 + (null_pc1 >= obs_pc1[s]).sum()) / (1 + args.n_rep)
        print(f"  {s}: 观测 PC1 = {obs_pc1[s]:.3f}   p = {p:.3f}"
              f"   {'超出零参照' if p < 0.05 else '在零参照范围内'}")

    # ---------- (2) 少数簇是否超出无簇模型 ----------
    print("\n" + "=" * 78)
    print(">>> (2) 少数簇的规模/间隙/同向性，是否超出\"无簇但同形状\"的零参照")
    print("=" * 78)
    mu = np.mean([X[s].mean(axis=0) for s in seeds], axis=0)
    cov = np.mean([np.cov(X[s], rowvar=False) for s in seeds], axis=0)
    null_rows = []
    for _ in range(args.n_rep):
        r = describe_minority(rng.multivariate_normal(mu, cov, size=n))
        if r:
            null_rows.append(r)
    N = pd.DataFrame(null_rows)
    print(f"  零参照（匹配均值协方差的高斯，无簇）{len(N)}/{args.n_rep} 次产出了少数簇：")
    for col, label in [("size", "少数簇规模"), ("gap_ratio", "密度间隙比"),
                       ("cos_members", "成员方向余弦")]:
        print(f"    {label:<12} 中位 {N[col].median():.2f}   "
              f"5%--95% 分位 [{N[col].quantile(.05):.2f}, {N[col].quantile(.95):.2f}]")

    print("\n  观测值与零参照的比较：")
    obs_rows = []
    for s in seeds:
        r = describe_minority(X[s])
        ps = {c: (1 + (N[c] >= r[c]).sum()) / (1 + len(N))
              for c in ("size", "gap_ratio", "cos_members")}
        print(f"    {s}: 规模 {r['size']}(p={ps['size']:.3f})  "
              f"间隙比 {r['gap_ratio']:.2f}(p={ps['gap_ratio']:.3f})  "
              f"方向余弦 {r['cos_members']:+.3f}(p={ps['cos_members']:.3f})")
        obs_rows.append({"seed": s, **r, **{f"p_{k}": v for k, v in ps.items()}})
    print("\n  p 小 = 观测值超出无簇模型的预期；p 不小 = 无簇的同形状高斯就能产出这样的少数簇")

    pd.DataFrame(obs_rows).to_csv(os.path.join(out_dir, "observed_vs_null.csv"), index=False)
    N.to_csv(os.path.join(out_dir, "null_minority_distribution.csv"), index=False)
    pd.DataFrame({"null_pc1_share": null_pc1}).to_csv(
        os.path.join(out_dir, "null_pc1_isotropic.csv"), index=False)
    print(f"\n>>> 已保存到 {out_dir}")


if __name__ == "__main__":
    main()
