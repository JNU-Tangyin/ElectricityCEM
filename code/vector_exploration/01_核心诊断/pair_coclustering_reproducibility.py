# -*- coding: utf-8 -*-
"""
比较三次重训中家庭对的同簇频率。每次重训汇总多个 t-SNE/UMAP 参数设置，
再计算跨重训 Spearman 相关，并以家庭标签置换构造零分布。家庭对相互依赖，
因此不使用 Spearman 常规 p 值；三个 seed 对的 p 值仍需作多重比较校正。
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
from scipy.stats import spearmanr
from sklearn.cluster import HDBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

SEED_ROOT = paths.SEED_ROOT
OUT_DIR = None  # 默认经 paths.shape_out 路由到实验目录
PARAMS = [5, 15, 30]          # t-SNE 的 perplexity / UMAP 的 n_neighbors
EMBED_SEED = 42               # 降维本身的随机种子，与训练种子无关
N_PERM = 500


def load_seeds():
    dirs = sorted(d for d in os.listdir(SEED_ROOT)
                  if glob.glob(os.path.join(SEED_ROOT, d, "*_expert.npy")))
    out = {}
    for d in dirs:
        files = sorted(glob.glob(os.path.join(SEED_ROOT, d, "*_expert.npy")))
        ids = [os.path.basename(f).replace("_expert.npy", "") for f in files]
        out[d] = (ids, np.array([np.load(f) for f in files]))
    return out


def cofreq_vector(x_std, pairs):
    """该次重训内部、跨 2 种降维方法 × 3 个参数设置的同簇频率。"""
    hit = np.zeros(len(pairs))
    tot = np.zeros(len(pairs))
    for method in ("t-SNE", "UMAP"):
        for p in PARAMS:
            if method == "t-SNE":
                emb = TSNE(n_components=2, perplexity=p, random_state=EMBED_SEED,
                           init="pca").fit_transform(x_std)
            else:
                emb = umap.UMAP(n_neighbors=p, n_components=2,
                                random_state=EMBED_SEED).fit_transform(x_std)
            lab = HDBSCAN(min_cluster_size=4, min_samples=2).fit_predict(emb)
            for k, (i, j) in enumerate(pairs):
                if lab[i] == -1 or lab[j] == -1:
                    continue
                tot[k] += 1
                hit[k] += lab[i] == lab[j]
    return np.where(tot > 0, hit / np.maximum(tot, 1), np.nan)


def permutation_test(a, b, pairs, pos, n_house, rng, n_perm):
    ok = np.isfinite(a) & np.isfinite(b)
    obs = spearmanr(a[ok], b[ok])[0]
    null = np.empty(n_perm)
    for t in range(n_perm):
        perm = rng.permutation(n_house)
        b_shuf = np.empty(len(pairs))
        for k, (i, j) in enumerate(pairs):
            pi, pj = perm[i], perm[j]
            b_shuf[k] = b[pos[(pi, pj) if pi < pj else (pj, pi)]]
        m = np.isfinite(a) & np.isfinite(b_shuf)
        null[t] = spearmanr(a[m], b_shuf[m])[0]
    return obs, null


def holm_adjust(p_values):
    """Holm correction for the three pairwise seed comparisons."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(
        1.0, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    return adjusted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default=OUT_DIR)
    ap.add_argument("--n_perm", type=int, default=N_PERM)
    args = ap.parse_args()
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    data = load_seeds()
    seeds = list(data)
    ids = data[seeds[0]][0]
    n = len(ids)
    for s in seeds:
        assert data[s][0] == ids, f"{s} 的家庭清单与其它重训不一致"

    keep = paths.dims_geometry(data[seeds[0]][1].shape[1])
    # 单一标准化器：各自拟合会把三次重训放进不同坐标尺度，而跨重训比较正是本检验的内容
    scaler = StandardScaler().fit(np.vstack([data[s][1][:, keep] for s in seeds]))

    pairs = list(itertools.combinations(range(n), 2))
    pos = {(i, j): k for k, (i, j) in enumerate(pairs)}
    print(f">>> {n} 户 × {len(seeds)} 次重训；{len(pairs)} 个家庭对 × "
          f"{2 * len(PARAMS)} 个嵌入设置\n")

    vecs = {}
    for s in seeds:
        vecs[s] = cofreq_vector(scaler.transform(data[s][1][:, keep]), pairs)
        v = vecs[s][np.isfinite(vecs[s])]
        print(f"    {s}: 同簇频率 mean={v.mean():.3f} "
              f"（=0 的对 {np.mean(v == 0):.1%}，=1 的对 {np.mean(v == 1):.1%}）")

    pd.DataFrame({"h1": [ids[i] for i, _ in pairs],
                  "h2": [ids[j] for _, j in pairs],
                  **{f"cofreq_{s}": vecs[s] for s in seeds}}
                 ).to_csv((os.path.join(args.output_dir, "pair_cofreq.csv") if args.output_dir else paths.shape_out("pair_cofreq.csv")), index=False)

    rng = np.random.default_rng(0)
    rows = []
    print(f"\n>>> 跨重训相关性（置换 {args.n_perm} 次）")
    print(f"{'比较':<22}{'观测rho':>9}{'零均值':>9}{'零p95':>9}{'置换p':>9}{'朴素p':>11}")
    for i, j in itertools.combinations(range(len(seeds)), 2):
        a, b = vecs[seeds[i]], vecs[seeds[j]]
        obs, null = permutation_test(a, b, pairs, pos, n, rng, args.n_perm)
        m = np.isfinite(a) & np.isfinite(b)
        naive_p = spearmanr(a[m], b[m])[1]
        perm_p = (1 + np.sum(null >= obs)) / (1 + args.n_perm)
        rows.append({"seed_a": seeds[i], "seed_b": seeds[j], "rho_obs": obs,
                     "null_mean": null.mean(), "null_q95": np.quantile(null, 0.95),
                     "p_perm": perm_p, "p_naive_invalid": naive_p,
                     "n_pairs": int(m.sum())})
        print(f"{seeds[i][-2:] + ' vs ' + seeds[j][-2:]:<22}{obs:>+9.3f}"
              f"{null.mean():>+9.3f}{np.quantile(null, 0.95):>+9.3f}"
              f"{perm_p:>9.4f}{naive_p:>11.1e}")

    df = pd.DataFrame(rows)
    df["p_holm"] = holm_adjust(df["p_perm"].to_numpy())
    print("\n>>> Holm 校正后的三次 seed 对比较")
    print(df[["seed_a", "seed_b", "p_perm", "p_holm"]].to_string(index=False))
    out = (os.path.join(args.output_dir, "pair_coclustering_reproducibility.csv") if args.output_dir else paths.shape_out("pair_coclustering_reproducibility.csv"))
    df.to_csv(out, index=False)
    print(f"\n>>> 已保存: {out}")
    print("    朴素p一列仅用于说明它为何不可用：2485个对共享家庭、并不独立。")


if __name__ == "__main__":
    main()
