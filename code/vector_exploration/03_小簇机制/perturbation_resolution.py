# -*- coding: utf-8 -*-
"""
比较沿观察到的尾部方向与随机方向扰动策略时的收益变化。使用不重叠的 72 小时窗口
估计窗口间变异；287 小时实际提供 3 个完整窗口。逐户结果仅作描述，不能视为显著性检验。
总体比较以家庭为配对单位；方向由同一批向量事后选出，结果不代表独立测试期上的泛化效果。
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_基础"))
import paths
sys.path.insert(0, paths.CODE)
import argparse
import glob
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

import energy_storage_env as ese                                   # noqa: E402
from energy_storage_env import EnergyStorageEnv                     # noqa: E402


warnings.filterwarnings("ignore")

DATA_DIR = paths.DATA
SEED_ROOT = paths.SEED_ROOT
OUT_DIR = os.path.join(paths.SHAPE, "exp25_perturbation_resolution")
WINDOW_H = 72
SELL_RATIO = 1.0
N_RANDOM = 20
DELTA_FACTOR = 1.0          # 相对于"成员质心 - 主体质心"的实际幅度


def tail_direction(Z):
    lab = HDBSCAN(min_cluster_size=4, min_samples=2).fit_predict(Z)
    real = [c for c in set(lab) if c != -1]
    sm = min(real, key=lambda c: (lab == c).sum())
    mem = np.where(lab == sm)[0]
    oth = np.setdiff1d(np.arange(len(Z)), mem)
    d = Z[mem].mean(0) - Z[oth].mean(0)
    return d / np.linalg.norm(d), np.linalg.norm(d)


def profit(df, theta, start):
    env = EnergyStorageEnv(df, episode_length_hours=WINDOW_H, sell_price_ratio=SELL_RATIO)
    obs, _ = env.reset(options={"start_idx": start})
    w, b, tot = theta[:35], theta[35], 0.0
    for _ in range(WINDOW_H):
        a = float(np.clip(np.tanh(obs @ w + b), -1.0, 1.0))
        obs, _, done, _, info = env.step(np.array([a], dtype=np.float64))
        tot += info["profit"]
        if done:
            break
    env.close()
    return tot


def main():
    ap = argparse.ArgumentParser(description="扰动效应的可分辨性")
    ap.add_argument("--output_dir", default=OUT_DIR)
    ap.add_argument("--n_random", type=int, default=N_RANDOM)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    seeds = ["cem_seed0", "cem_seed1"]
    ids = sorted(os.path.basename(f).replace("_expert.npy", "")
                 for f in glob.glob(os.path.join(SEED_ROOT, seeds[0], "*_expert.npy")))
    raw = {s: np.array([np.load(os.path.join(SEED_ROOT, s, f"{h}_expert.npy")) for h in ids])
           for s in seeds}
    scaler = StandardScaler().fit(np.vstack([raw[s] for s in seeds]))
    Z = {s: scaler.transform(raw[s]) for s in seeds}
    W = {s: tail_direction(Z[s]) for s in seeds}
    scale = scaler.scale_

    rows = []
    for k, hid in enumerate(ids, 1):
        df = pd.read_csv(os.path.join(DATA_DIR, f"{hid}.csv"), index_col=0, parse_dates=True)
        # 互不重叠的窗口，这是 287 小时能提供的全部独立测量
        starts = [s for s in range(0, len(df) - WINDOW_H + 1, WINDOW_H)]
        for s in seeds:
            th = raw[s][ids.index(hid)]
            wdir, mag = W[s]
            dt = DELTA_FACTOR * mag * wdir * scale
            rdirs = []
            for _ in range(args.n_random):
                r = rng.standard_normal(len(wdir))
                rdirs.append(DELTA_FACTOR * mag * (r / np.linalg.norm(r)) * scale)
            for st in starts:
                base = profit(df, th, st)
                tail = np.mean([abs(profit(df, th + dt, st) - base),
                                abs(profit(df, th - dt, st) - base)])
                rand = np.mean([abs(profit(df, th + r, st) - base) for r in rdirs])
                rows.append(dict(household=hid, seed=s, start=st, base=base,
                                 d_tail=tail, d_random=rand))
        if k % 20 == 0 or k == len(ids):
            print(f"    {k}/{len(ids)} 户", flush=True)

    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(args.output_dir, "per_window_perturbation.csv"), index=False)

    print(f"\n每户独立窗口数: {D.groupby(['household','seed']).size().median():.0f}")
    print("=" * 74)
    print(">>> 逐户层面：该户自己的 4 个窗口够不够分辨")
    print("=" * 74)
    g = D.groupby(["household", "seed"])
    per = g.apply(lambda d: pd.Series({
        "mean_diff": (d.d_random - d.d_tail).mean(),
        "se_diff": (d.d_random - d.d_tail).std(ddof=1) / np.sqrt(len(d)),
    }), include_groups=False).reset_index()
    per["resolved"] = per.mean_diff.abs() > 2 * per.se_diff
    print(f"  '随机更陡'这一差别在该户自身窗口上可分辨的比例: "
          f"{per.resolved.mean():.1%} ({int(per.resolved.sum())}/{len(per)})")

    print("\n" + "=" * 74)
    print(">>> 总体层面：把 71 户的配对差值当独立样本")
    print("=" * 74)
    hh = D.groupby(["household", "seed"])[["d_tail", "d_random"]].mean().reset_index()
    for s in seeds:
        x = hh[hh.seed == s]
        diff = x.d_random - x.d_tail
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        ratio = x.d_tail.median() / x.d_random.median()
        print(f"  {s}: 尾巴/随机 中位比 {ratio:.3f}   "
              f"配对差均值 {diff.mean():+.3f} ± {se:.3f} (SE)   "
              f"t = {diff.mean()/se:+.1f}   尾巴更平的户数 {int((diff>0).sum())}/{len(diff)}")
    print(f"\n>>> 已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
