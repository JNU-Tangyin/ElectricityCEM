# -*- coding: utf-8 -*-
"""
对观察到的 HDBSCAN 小簇作事后诊断：比较原空间与 UMAP 结果、密度间隙、参数敏感性和
跨重训位移。分析均依赖数据驱动的簇定义，不构成独立的簇存在性检验。

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
import umap
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import HDBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

MIN_CLUSTER_SIZE = 4
MIN_SAMPLES = 2
EMBED_SEED = 42
MCS_SWEEP = [3, 4, 5, 8, 10, 15]


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


def minority(labels):
    """最小的真实簇的成员下标；无真实簇时返回空数组。"""
    real = [c for c in set(labels) if c != -1]
    if not real:
        return np.array([], dtype=int)
    smallest = min(real, key=lambda c: (labels == c).sum())
    return np.where(labels == smallest)[0]


def sizes(labels):
    return sorted((int((labels == c).sum()) for c in set(labels) if c != -1), reverse=True)


def internal_ratio(D, members):
    """成员两两平均距离 / 全体两两平均距离。"""
    iu = np.triu_indices(D.shape[0], 1)
    within = [D[a, b] for a, b in itertools.combinations(members, 2)]
    return np.mean(within) / D[iu].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()
    out_dir = args.output_dir or os.path.join(
        paths.SHAPE, "exp19_small_cluster_density_gap")
    os.makedirs(out_dir, exist_ok=True)

    ids, seeds, X = load_panel()
    n, d = len(ids), X[seeds[0]].shape[1]
    print(f">>> {n} 户 × {len(seeds)} 重训 × {d} 维\n")

    # ---------- (1) 几何空间直接聚类 vs 降维后聚类 ----------
    print("=" * 78)
    print(">>> (1) 直接在原空间聚类 vs 降维后聚类：分割是否相同")
    print("=" * 78)
    rows1, members36 = [], {}
    for s in seeds:
        D36 = squareform(pdist(X[s]))
        lab36 = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE,
                        min_samples=MIN_SAMPLES).fit_predict(X[s])
        m36 = minority(lab36)
        members36[s] = [ids[i] for i in m36]
        emb = umap.UMAP(n_neighbors=15, n_components=2,
                        random_state=EMBED_SEED).fit_transform(X[s])
        lab2 = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE,
                       min_samples=MIN_SAMPLES).fit_predict(emb)
        m2 = minority(lab2)
        D2 = squareform(pdist(emb))
        print(f"  {s}")
        print(f"    几何空间直接聚类: 簇规模 {sizes(lab36)}  噪声 {int((lab36 == -1).sum())}")
        print(f"    UMAP 后聚类   : 簇规模 {sizes(lab2)}  噪声 {int((lab2 == -1).sum())}")
        print(f"    少数簇成员（原空间）: {sorted(members36[s])}")
        r36 = internal_ratio(D36, m36) if len(m36) > 1 else np.nan
        r2 = internal_ratio(D2, m2) if len(m2) > 1 else np.nan
        print(f"    成员内部平均距离/全体平均: 原空间 {r36:.2f}，二维嵌入 {r2:.2f}"
              f"  -> 投影压缩 {r36 / r2:.1f} 倍")
        rows1.append({"seed": s, "sizes_native": str(sizes(lab36)),
                      "noise_native": int((lab36 == -1).sum()),
                      "sizes_umap": str(sizes(lab2)), "noise_umap": int((lab2 == -1).sum()),
                      "minority_native": ";".join(sorted(members36[s])),
                      "internal_ratio_native": r36, "internal_ratio_umap": r2})
    common = set.intersection(*(set(v) for v in members36.values()))
    print(f"\n  三次重训少数簇成员的交集: {sorted(common) or '空集'}")

    # ---------- (2) 密度间隙 ----------
    print("\n" + "=" * 78)
    print(">>> (2) 密度间隙：为什么不被判为噪声，以及它是否比主体致密")
    print("=" * 78)
    rows2 = []
    for s in seeds:
        lab = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE,
                      min_samples=MIN_SAMPLES).fit_predict(X[s])
        mem = minority(lab)
        oth = np.setdiff1d(np.arange(n), mem)
        D = squareform(pdist(X[s]))
        np.fill_diagonal(D, np.inf)
        nn_in = np.mean([D[a, mem[mem != a]].min() for a in mem])
        nn_out = np.mean([D[a, oth].min() for a in mem])
        nn_bulk = np.mean([D[a, oth[oth != a]].min() for a in oth])
        print(f"  {s}（{len(mem)} 户）: 到最近同伴 {nn_in:.2f}   到最近非成员 {nn_out:.2f}"
              f"  比值 {nn_out / nn_in:.2f}   主体内部最近邻 {nn_bulk:.2f}")
        rows2.append({"seed": s, "n_members": len(mem), "nn_within": nn_in,
                      "nn_to_nonmember": nn_out, "ratio": nn_out / nn_in,
                      "nn_bulk": nn_bulk, "gap_width": nn_out - nn_in})
    print("    比值 >1 = 存在密度间隙，故不被判为噪声；"
          "到最近同伴 > 主体内部最近邻 = 该簇比主体稀疏，是边缘群而非致密簇")

    # ---------- (3) min_cluster_size 敏感性 ----------
    print("\n" + "=" * 78)
    print(">>> (3) min_cluster_size 敏感性（UMAP n_neighbors=15）")
    print("=" * 78)
    rows3 = []
    for s in seeds:
        emb = umap.UMAP(n_neighbors=15, n_components=2,
                        random_state=EMBED_SEED).fit_transform(X[s])
        for mcs in MCS_SWEEP:
            lab = HDBSCAN(min_cluster_size=mcs, min_samples=MIN_SAMPLES).fit_predict(emb)
            rows3.append({"seed": s, "min_cluster_size": mcs, "sizes": str(sizes(lab)),
                          "noise": int((lab == -1).sum())})
        sub = [r for r in rows3 if r["seed"] == s]
        print(f"  {s}: " + " | ".join(f"mcs={r['min_cluster_size']}:{r['sizes']}"
                                      f"(噪声{r['noise']})" for r in sub))

    # ---------- (4) 间隙宽度 vs 重训位移，逐户稳定性 ----------
    print("\n" + "=" * 78)
    print(">>> (4) 成员为何跨重训变化：间隙宽度 vs 重训位移")
    print("=" * 78)
    disp = np.mean([np.linalg.norm(X[a][i] - X[b][i]) for i in range(n)
                    for a, b in itertools.combinations(seeds, 2)])
    gap = np.mean([r["gap_width"] for r in rows2])
    print(f"  同一户在两次重训之间的位移: {disp:.2f}")
    print(f"  少数簇与主体之间的间隙宽度: {gap:.2f}")
    print(f"  ==> 位移是间隙宽度的 {disp / gap:.1f} 倍，故边界每次重画")

    # 边界 = 该次少数簇成员中离质心最近者的距离
    cdist = {s: np.linalg.norm(X[s] - X[s].mean(axis=0), axis=1) for s in seeds}
    thr = {}
    for s in seeds:
        lab = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE,
                      min_samples=MIN_SAMPLES).fit_predict(X[s])
        thr[s] = cdist[s][minority(lab)].min()
    sd_all = np.array([np.std([cdist[s][i] for s in seeds]) for i in range(n)])
    print(f"\n  各重训的边界值: " + "  ".join(f"{s[-2:]}={thr[s]:.2f}" for s in seeds))
    print(f"  全体到质心距离的中位数: {np.median([cdist[s] for s in seeds]):.2f}")
    print(f"  单户到质心距离的跨重训标准差（全体中位）: {np.median(sd_all):.2f}")

    ever = sorted({h for v in members36.values() for h in v},
                  key=lambda h: -sum(1 for s in seeds if h in members36[s]))
    print(f"\n  曾进入少数簇的 {len(ever)} 户：持久位置有多靠外 vs 跨重训波动多大")
    print(f"  {'家庭':<8}{'到质心距离(逐重训)':>24}{'均值':>7}{'标准差':>8}{'≥边界':>7}")
    rows4 = []
    for h in ever:
        i = ids.index(h)
        v = [cdist[s][i] for s in seeds]
        cnt = sum(1 for s in seeds if cdist[s][i] >= thr[s])
        print(f"  {h:<8}{'  '.join(f'{x:5.2f}' for x in v):>24}"
              f"{np.mean(v):>7.2f}{np.std(v):>8.2f}{cnt:>5}/{len(seeds)}")
        rows4.append({"household": h, **{f"centroid_dist_{s}": cdist[s][i] for s in seeds},
                      "mean": np.mean(v), "sd_across_seeds": np.std(v),
                      "n_seeds_beyond_boundary": cnt,
                      "sd_vs_population_median": np.std(v) / np.median(sd_all)})
    print("    标准差远小于全体中位的家庭才能每次都在簇内；其余在边界两侧摆动")

    for name, rows in [("native_vs_umap", rows1), ("density_gap", rows2),
                       ("min_cluster_size_sweep", rows3), ("per_household_stability", rows4)]:
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    pd.DataFrame([{"mean_retraining_displacement": disp, "mean_gap_width": gap,
                   "displacement_over_gap": disp / gap,
                   "median_centroid_dist": float(np.median([cdist[s] for s in seeds])),
                   "median_sd_across_seeds": float(np.median(sd_all))}]
                 ).to_csv(os.path.join(out_dir, "displacement_vs_gap.csv"), index=False)
    print(f"\n>>> 已保存到 {out_dir}")


if __name__ == "__main__":
    main()
