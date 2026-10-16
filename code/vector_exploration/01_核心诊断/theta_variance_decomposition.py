# -*- coding: utf-8 -*-
"""
估计 71 户 × 3 次重训的户间/户内方差，报告逐维与汇总 ICC，并以 34 维几何坐标
进行户内/户间距离置换检验。三次重训不足以精确分离随机噪声与 seed 特有偏移，
因此 ICC 应视为不确定的描述性估计。
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_基础"))
import paths
import argparse
import glob
import itertools

import numpy as np

import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import f as f_dist


N_PERM = 2000

# 原始 36 维 -> θ 语义组（按 EnergyStorageEnv._get_obs() 的观测顺序）
GROUPS = {
    0: "soc_w", 1: "pv_w", 2: "load_w", 3: "(location)",
    4: "time_w", 5: "time_w", 6: "price_now_w", 7: "price_now_w",
    **{d: "price_near_w" for d in range(8, 14)},
    **{d: "price_mid_w" for d in range(14, 20)},
    **{d: "price_far_w" for d in range(20, 32)},
    **{d: "weather_w" for d in range(32, 35)},
    35: "(bias)",
}


def load_panel():
    """返回 (household_ids, seed_names, theta)，theta 形状为 (n_house, n_seed, 36)，
    共享标准化器在全部 n_house*n_seed 行上拟合。"""
    seeds = sorted(d for d in os.listdir(paths.SEED_ROOT)
                   if glob.glob(os.path.join(paths.SEED_ROOT, d, "*_expert.npy")))
    per_seed = {}
    for s in seeds:
        files = sorted(glob.glob(os.path.join(paths.SEED_ROOT, s, "*_expert.npy")))
        ids = [os.path.basename(f).replace("_expert.npy", "") for f in files]
        per_seed[s] = (ids, np.array([np.load(f) for f in files]))
    ids = per_seed[seeds[0]][0]
    for s in seeds:
        assert per_seed[s][0] == ids, f"{s} 的家庭清单与其它重训不一致"

    keep = paths.dims_all(per_seed[seeds[0]][1].shape[1])
    stack = np.stack([per_seed[s][1][:, keep] for s in seeds], axis=1)  # (h, s, d)
    flat = stack.reshape(-1, stack.shape[2])
    stack = ((stack - flat.mean(axis=0)) / flat.std(axis=0, ddof=0))
    return ids, seeds, stack, [GROUPS[d] for d in keep]


def icc1(y, alpha=0.05):
    """单因素随机效应 ICC(1)。y 形状 (n_groups, k)，每组观测数相同。"""
    n, k = y.shape
    gm = y.mean()
    group_means = y.mean(axis=1)
    ms_b = k * ((group_means - gm) ** 2).sum() / (n - 1)
    ms_w = ((y - group_means[:, None]) ** 2).sum() / (n * (k - 1))
    var_a = (ms_b - ms_w) / k                      # 可能为负：户间方差估计为 0 的抽样表现
    icc = var_a / (var_a + ms_w) if (var_a + ms_w) != 0 else np.nan
    # F 分布置信区间（Shrout & Fleiss 的 ICC(1,1) 形式）
    F = ms_b / ms_w if ms_w > 0 else np.nan
    df1, df2 = n - 1, n * (k - 1)
    fl = F / f_dist.ppf(1 - alpha / 2, df1, df2)
    fu = F * f_dist.ppf(1 - alpha / 2, df2, df1)
    return {"icc": icc, "ci_low": (fl - 1) / (fl + k - 1), "ci_high": (fu - 1) / (fu + k - 1),
            "ms_between": ms_b, "ms_within": ms_w, "var_between": var_a, "var_within": ms_w,
            "F": F}


def distance_test(stack, rng, n_perm=N_PERM):
    """户内 vs 户间平均两两距离，置换检验打乱"这一次运行属于哪一户"。

    距离在几何坐标上算：stack 为逐维 ICC 保留了全部 36 维，但 location 与 bias
    的拆分不可识别，计入距离只会加噪声。"""
    h, s, d = stack.shape
    X = stack[:, :, paths.dims_geometry(d)].reshape(h * s, -1)
    labels = np.repeat(np.arange(h), s)
    D = squareform(pdist(X))
    iu = np.triu_indices(h * s, k=1)
    dist = D[iu]
    same = labels[iu[0]] == labels[iu[1]]

    obs = dist[same].mean() - dist[~same].mean()
    null = np.empty(n_perm)
    for t in range(n_perm):
        perm = rng.permutation(labels)
        sm = perm[iu[0]] == perm[iu[1]]
        null[t] = dist[sm].mean() - dist[~sm].mean()
    # 单侧：假设是"户内更近"，即差值更负
    p = (1 + np.sum(null <= obs)) / (1 + n_perm)
    return {"within_mean": dist[same].mean(), "between_mean": dist[~same].mean(),
            "ratio": dist[same].mean() / dist[~same].mean(), "diff_obs": obs,
            "null_mean": null.mean(), "null_q05": np.quantile(null, 0.05), "p_perm": p,
            "n_within_pairs": int(same.sum()), "n_between_pairs": int((~same).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--n_perm", type=int, default=N_PERM)
    args = ap.parse_args()
    out_dir = args.output_dir or os.path.join(
        paths.SHAPE, "exp1b_variance_decomposition")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    ids, seeds, stack, dim_groups = load_panel()
    h, s, d = stack.shape
    print(f">>> {h} 户 × {s} 次重训 × {d} 维（共享标准化器）")
    print(f">>> 重训: {', '.join(seeds)}\n")

    # ---------- (1)(2) 逐维与汇总 ICC ----------
    rows = [{"dim_index": j, "theta_group": dim_groups[j], **icc1(stack[:, :, j])}
            for j in range(d)]
    per_dim = pd.DataFrame(rows)

    vb_raw, vw = per_dim.var_between.sum(), per_dim.var_within.sum()
    vb_clip = per_dim.var_between.clip(lower=0).sum()
    icc_raw = vb_raw / (vb_raw + vw)
    icc_clip = vb_clip / (vb_clip + vw)

    print("=" * 74)
    print(">>> (1) 汇总 ICC：θ 方差中属于户间的比例")
    print("=" * 74)
    print(f"    不截断负分量   ICC = {icc_raw:.3f}   (户间 {vb_raw:.2f} / 户内 {vw:.2f})")
    print(f"    截断负分量为0  ICC = {icc_clip:.3f}  (上偏，仅作上界参考)")
    print(f"    逐维 ICC 中位数 {per_dim.icc.median():.3f}，"
          f"负分量维数 {(per_dim.var_between < 0).sum()}/{d}")

    # ---------- (4) 按语义组 ----------
    print("\n" + "=" * 74)
    print(">>> (2) 按 θ 语义组：家庭信号是否集中在少数几维")
    print("=" * 74)
    agg = (per_dim.groupby("theta_group")
           .apply(lambda g: pd.Series({
               "n_dims": len(g),
               "icc_group": g.var_between.sum() / (g.var_between.sum() + g.var_within.sum()),
               "icc_dim_median": g.icc.median(),
               "icc_dim_max": g.icc.max()}), include_groups=False)
           .sort_values("icc_group", ascending=False))
    print(agg.round(3).to_string())

    print("\n>>> 逐维 ICC 前 8 名")
    print(per_dim.nlargest(8, "icc")[["dim_index", "theta_group", "icc", "ci_low", "ci_high"]]
          .round(3).to_string(index=False))

    # ---------- (3) 距离检验：全体 71 户 vs 分层 10 户 ----------
    print("\n" + "=" * 74)
    print(">>> (3) 距离版 户内/户间（置换检验，与 10 户那次同设计）")
    print("=" * 74)
    dist_rows = []
    full = distance_test(stack, rng, args.n_perm)
    dist_rows.append({"subset": f"all_{h}_households", **full})
    print(f"    全体 {h} 户: 户内 {full['within_mean']:.3f} vs 户间 "
          f"{full['between_mean']:.3f}  比值 {full['ratio']:.3f}  置换 p={full['p_perm']:.4f}")

    # 10 户子集：household_id 从 exp10 的产物读取，不在本脚本硬编码任何家庭名单
    sig = os.path.join(paths.SHAPE, "exp10_multiseed_signal", "household_signal_summary.csv")
    if os.path.exists(sig):
        sub_ids = pd.read_csv(sig, dtype={"household": str})["household"].tolist()
        idx = [i for i, hid in enumerate(ids) if hid in set(sub_ids)]
        if len(idx) >= 3:
            sub = distance_test(stack[idx], rng, args.n_perm)
            dist_rows.append({"subset": f"stratified_{len(idx)}_households", **sub})
            print(f"    分层 {len(idx)} 户: 户内 {sub['within_mean']:.3f} vs 户间 "
                  f"{sub['between_mean']:.3f}  比值 {sub['ratio']:.3f}  "
                  f"置换 p={sub['p_perm']:.4f}")
            print(f"    -> 分层抽样使比值从 {full['ratio']:.3f} 变为 {sub['ratio']:.3f}")
    else:
        print(f"    (未找到 {sig}，跳过分层子集对比)")

    per_dim.to_csv(os.path.join(out_dir, "icc_per_dimension.csv"), index=False)
    agg.reset_index().to_csv(os.path.join(out_dir, "icc_by_theta_group.csv"), index=False)
    pd.DataFrame([{"n_households": h, "n_seeds": s, "n_dims": d,
                   "icc_overall_raw": icc_raw, "icc_overall_clipped": icc_clip,
                   "var_between_sum": vb_raw, "var_within_sum": vw,
                   "n_dims_negative_component": int((per_dim.var_between < 0).sum())}]
                 ).to_csv(os.path.join(out_dir, "icc_overall.csv"), index=False)
    pd.DataFrame(dist_rows).to_csv(
        os.path.join(out_dir, "within_between_distance.csv"), index=False)
    print(f"\n>>> 已保存到 {out_dir}")
    print(">>> 限制：k=3 下 ICC 抽样误差大，区间很宽；3 个 seed 无法把重训噪声与"
          "seed 特有的系统偏移分开（后者会被算进户内方差，低估 ICC）。")


if __name__ == "__main__":
    main()
