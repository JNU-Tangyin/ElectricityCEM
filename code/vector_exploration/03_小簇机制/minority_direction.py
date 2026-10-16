# -*- coding: utf-8 -*-
"""
描述 HDBSCAN 最小非噪声簇的质心差方向、跨重训方向一致性及维度贡献。
簇和方向均由同一批向量选出，结果属于事后探索；成员恢复率不代表独立分类性能。
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
from scipy.stats import spearmanr
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

MIN_CLUSTER_SIZE = 4
MIN_SAMPLES = 2
N_REP = 300
TOP_K = [1, 2, 3, 5, 8, 12, 20, 36]

DIM_NAMES = (["soc", "pv_gen", "user_load", "location", "hour_sin", "hour_cos",
              "price_global", "price_now"]
             + [f"price_t+{i}" for i in range(1, 25)]
             + ["cloud_cover", "radiation", "radiation_t+3", "bias"])


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
    names = [DIM_NAMES[j] for j in keep] if len(DIM_NAMES) >= max(keep) + 1 else \
        [f"dim{j}" for j in keep]
    return ids, seeds, {s: scaler.transform(per[s][1][:, keep]) for s in seeds}, names


def minority_direction(M):
    """返回 (成员下标, 单位方向向量)；方向 = 成员质心 - 主体质心。"""
    lab = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES).fit_predict(M)
    real = [c for c in set(lab) if c != -1]
    if not real:
        return None, None
    sm = min(real, key=lambda c: (lab == c).sum())
    mem = np.where(lab == sm)[0]
    oth = np.setdiff1d(np.arange(len(M)), mem)
    if len(mem) < 3 or len(oth) < 3:
        return None, None
    w = M[mem].mean(axis=0) - M[oth].mean(axis=0)
    nrm = np.linalg.norm(w)
    return mem, (w / nrm if nrm > 0 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_rep", type=int, default=N_REP)
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()
    out_dir = args.output_dir or os.path.join(
        paths.SHAPE, "exp21_minority_direction")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    ids, seeds, X, names = load_panel()
    n, d = len(ids), X[seeds[0]].shape[1]
    print(f">>> {n} 户 × {len(seeds)} 重训 × {d} 维\n")

    # ---------- (1) 逐重训方向及两两余弦 ----------
    W, MEM = {}, {}
    for s in seeds:
        MEM[s], W[s] = minority_direction(X[s])
    print("=" * 76)
    print(">>> (1) 分离方向的跨重训一致性")
    print("=" * 76)
    obs_cos = {}
    for a, b in itertools.combinations(seeds, 2):
        c = float(W[a] @ W[b])
        obs_cos[(a, b)] = c
        print(f"  {a[-2:]} vs {b[-2:]}: 方向余弦 {c:+.3f}"
              f"   (成员数 {len(MEM[a])} / {len(MEM[b])})")

    # ---------- (2) 零参照 ----------
    print("\n" + "=" * 76)
    print(">>> (2) 零参照：无簇高斯独立抽三份，方向余弦能达到多少")
    print("=" * 76)
    mu = np.mean([X[s].mean(axis=0) for s in seeds], axis=0)
    cov = np.mean([np.cov(X[s], rowvar=False) for s in seeds], axis=0)
    null_cos = []
    for _ in range(args.n_rep):
        ws = []
        for _ in range(2):
            _, w = minority_direction(rng.multivariate_normal(mu, cov, size=n))
            if w is not None:
                ws.append(w)
        if len(ws) == 2:
            null_cos.append(float(ws[0] @ ws[1]))
    null_cos = np.array(null_cos)
    print(f"  {len(null_cos)}/{args.n_rep} 次成功：余弦均值 {null_cos.mean():+.3f}，"
          f"95% 分位 {np.quantile(null_cos, 0.95):+.3f}，最大 {null_cos.max():+.3f}")
    rows2 = []
    for (a, b), c in obs_cos.items():
        p = (1 + (null_cos >= c).sum()) / (1 + len(null_cos))
        print(f"  观测 {a[-2:]} vs {b[-2:]} = {c:+.3f}   p = {p:.4f}"
              f"   {'超出零参照' if p < 0.05 else '在零参照范围内'}")
        rows2.append({"seed_a": a, "seed_b": b, "cos_obs": c, "p_vs_null": p,
                      "null_mean": null_cos.mean(), "null_q95": np.quantile(null_cos, .95)})

    # ---------- (3) 维度贡献排序 ----------
    print("\n" + "=" * 76)
    print(">>> (3) 各维在分离方向上的贡献（方向向量的分量，按三次均值绝对值排序）")
    print("=" * 76)
    Wm = np.array([W[s] for s in seeds])
    contrib = pd.DataFrame({"dim": np.arange(d), "name": names,
                            **{f"w_{s[-2:]}": W[s] for s in seeds}})
    contrib["mean_abs"] = np.abs(Wm).mean(axis=0)
    contrib["same_sign"] = (np.sign(Wm) == np.sign(Wm[0])).all(axis=0)
    contrib["var_share"] = contrib.mean_abs ** 2 / (contrib.mean_abs ** 2).sum()
    top = contrib.sort_values("mean_abs", ascending=False)
    print(top.head(14)[["dim", "name"] + [f"w_{s[-2:]}" for s in seeds]
                       + ["mean_abs", "var_share", "same_sign"]].round(3).to_string(index=False))
    print(f"\n  前 5 维占方向平方长度的 {top.var_share.head(5).sum():.0%}，"
          f"前 12 维占 {top.var_share.head(12).sum():.0%}")
    rhos = [spearmanr(np.abs(W[a]), np.abs(W[b]))[0] for a, b in itertools.combinations(seeds, 2)]
    print(f"  贡献排序的跨重训一致性（|分量| 的 Spearman）: "
          + ", ".join(f"{r:+.3f}" for r in rhos))
    print(f"  三次符号一致的维数: {int(contrib.same_sign.sum())}/{d}")

    # ---------- (4) 稀疏恢复 ----------
    print("\n" + "=" * 76)
    print(">>> (4) 只用贡献最大的前 k 维，能否复原成员名单")
    print("=" * 76)
    rows4 = []
    for s in seeds:
        order = np.argsort(-np.abs(W[s]))
        line = []
        for k in TOP_K:
            sel = order[:k]
            score = X[s][:, sel] @ W[s][sel]
            pred = set(np.argsort(-score)[:len(MEM[s])])
            hit = len(pred & set(MEM[s]))
            line.append(f"k={k}:{hit}/{len(MEM[s])}")
            rows4.append({"seed": s, "k": k, "hit": hit, "n_members": len(MEM[s]),
                          "dims": ";".join(names[j] for j in sel)})
        print(f"  {s}: " + "  ".join(line))
    print("  （k=36 即用全部维度，命中数是该方法的上限）")

    contrib.to_csv(os.path.join(out_dir, "dimension_contribution.csv"), index=False)
    pd.DataFrame(rows2).to_csv(os.path.join(out_dir, "direction_cross_seed.csv"), index=False)
    pd.DataFrame(rows4).to_csv(os.path.join(out_dir, "sparse_recovery.csv"), index=False)
    pd.DataFrame({"null_cos": null_cos}).to_csv(
        os.path.join(out_dir, "null_direction_cos.csv"), index=False)
    print(f"\n>>> 已保存到 {out_dir}")


if __name__ == "__main__":
    main()
